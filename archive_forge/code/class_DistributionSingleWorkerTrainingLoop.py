import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils_v1 as dist_utils
from tensorflow.python.keras.engine import partial_batch_padding_handler as padding_util
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
class DistributionSingleWorkerTrainingLoop(training_utils_v1.TrainingLoop):
    """Training loop for distribution strategy with single worker."""

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, **kwargs):
        """Fit loop for Distribution Strategies."""
        dist_utils.validate_callbacks(input_callbacks=callbacks, optimizer=model.optimizer)
        dist_utils.validate_inputs(x, y)
        batch_size, steps_per_epoch = dist_utils.process_batch_and_step_size(model._distribution_strategy, x, batch_size, steps_per_epoch, ModeKeys.TRAIN, validation_split=validation_split)
        batch_size = model._validate_or_infer_batch_size(batch_size, steps_per_epoch, x)
        dataset = model._distribution_standardize_user_data(x, y, sample_weight=sample_weight, class_weight=class_weight, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle, epochs=epochs)
        if not dist_utils.is_distributing_by_cloning(model):
            with model._distribution_strategy.scope():
                dataset, _, _ = model._standardize_user_data(dataset, sample_weight=sample_weight, class_weight=class_weight, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle)
        val_dataset = None
        if validation_data:
            val_x, val_y, val_sample_weights = training_utils_v1.unpack_validation_data(validation_data)
            dist_utils.validate_inputs(val_x, val_y)
            _, validation_steps = dist_utils.process_batch_and_step_size(model._distribution_strategy, val_x, batch_size, validation_steps, ModeKeys.TEST)
            val_dataset = model._distribution_standardize_user_data(val_x, val_y, sample_weight=val_sample_weights, class_weight=None, batch_size=batch_size, validation_split=validation_split, shuffle=shuffle, allow_partial_batch=True)
        elif validation_split:
            raise ValueError('validation_split argument is not supported with distribution strategies.')
        if backend.is_tpu_strategy(model._distribution_strategy):
            steps_per_epoch = training_utils_v1.infer_steps_for_dataset(model, dataset, steps_per_epoch, epochs, steps_name='steps_per_epoch')
            if steps_per_epoch is None:
                raise ValueError('Number of steps could not be inferred from the data, please pass the steps_per_epoch argument.')
            if not context.executing_eagerly():
                return experimental_tpu_fit_loop(model, dataset, epochs=epochs, verbose=verbose, callbacks=callbacks, val_dataset=val_dataset, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq)
        return training_arrays_v1.fit_loop(model, dataset, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, val_inputs=val_dataset, shuffle=shuffle, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq, steps_name='steps_per_epoch')

    def evaluate(self, model, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, **kwargs):
        """Evaluate loop for Distribution Strategies."""
        dist_utils.validate_inputs(x, y)
        batch_size, steps = dist_utils.process_batch_and_step_size(model._distribution_strategy, x, batch_size, steps, ModeKeys.TEST)
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        dataset = model._distribution_standardize_user_data(x, y, sample_weight=sample_weight, batch_size=batch_size, allow_partial_batch=True)
        if backend.is_tpu_strategy(model._distribution_strategy):
            steps = training_utils_v1.infer_steps_for_dataset(model, dataset, steps, steps_name='steps')
            if steps is None:
                raise ValueError('Number of steps could not be inferred from the data, please pass the steps argument.')
            if not context.executing_eagerly():
                return experimental_tpu_test_loop(model, dataset, verbose=verbose, steps=steps, callbacks=callbacks)
        return training_arrays_v1.test_loop(model, inputs=dataset, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)

    def predict(self, model, x, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        """Predict loop for Distribution Strategies."""
        dist_utils.validate_inputs(x=x, y=None)
        batch_size, steps = dist_utils.process_batch_and_step_size(model._distribution_strategy, x, batch_size, steps, ModeKeys.PREDICT)
        batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
        dataset = model._distribution_standardize_user_data(x, batch_size=batch_size, allow_partial_batch=True)
        if backend.is_tpu_strategy(model._distribution_strategy):
            steps = training_utils_v1.infer_steps_for_dataset(model, dataset, steps, steps_name='steps')
            if steps is None:
                raise ValueError('Number of steps could not be inferred from the data, please pass the steps argument.')
            if not context.executing_eagerly():
                return experimental_tpu_predict_loop(model, dataset, verbose=verbose, steps=steps, callbacks=callbacks)
        return training_arrays_v1.predict_loop(model, dataset, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)
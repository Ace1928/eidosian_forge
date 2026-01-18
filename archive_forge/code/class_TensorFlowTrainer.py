import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
class TensorFlowTrainer(base_trainer.Trainer):

    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        if tf.distribute.has_strategy():
            self._distribute_strategy = tf.distribute.get_strategy()
        else:
            self._distribute_strategy = None
        self._distribute_reduction_method = None
        self._supports_reduce_retracing = Version(tf.__version__) >= Version('2.9.0')

    @property
    def distribute_strategy(self):
        return self._distribute_strategy or tf.distribute.get_strategy()

    @property
    def distribute_reduction_method(self):
        return self._distribute_reduction_method or 'auto'

    @distribute_reduction_method.setter
    def distribute_reduction_method(self, value):
        self._distribute_reduction_method = value

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
            self._loss_tracker.update_state(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn('The model does not have any trainable weights.')
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)
        self._loss_tracker.update_state(loss)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and (not force):
            return self.train_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            return self.train_step(data)
        if not self.run_eagerly:
            kwargs = {'jit_compile': self.jit_compile}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single training step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction=self.distribute_reduction_method)
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs
        if self.steps_per_execution > 1:
            train_function = multi_step_on_iterator
        else:
            train_function = one_step_on_iterator
        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            train_function = tf.function(train_function, **kwargs)
        self.train_function = train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and (not force):
            return self.test_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            return self.test_step(data)
        if not self.run_eagerly and self.jit_compile:
            kwargs = {'jit_compile': True}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_iterator(iterator):
            """Runs a single test step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction=self.distribute_reduction_method)
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            for _ in range(self.steps_per_execution):
                outputs = one_step_on_iterator(iterator)
            return outputs
        if self.steps_per_execution > 1:
            test_function = multi_step_on_iterator
        else:
            test_function = one_step_on_iterator
        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            test_function = tf.function(test_function, **kwargs)
        self.test_function = test_function

    def make_predict_function(self, force=False):
        if self.predict_function is not None and (not force):
            return self.predict_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)
        if not self.run_eagerly and self.jit_compile:
            kwargs = {'jit_compile': True}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            one_step_on_data = tf.function(one_step_on_data, **kwargs)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data_distributed(data):
            data = data[0]
            outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction=self.distribute_reduction_method)
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_data(data):
            outputs = one_step_on_data_distributed(data[:1])
            for single_step_data in data[1:]:
                step_outputs = one_step_on_data_distributed([single_step_data])
                outputs = tf.nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs, step_outputs)
            return outputs
        if self.steps_per_execution > 1:
            predict_function = multi_step_on_data
        else:
            predict_function = one_step_on_data_distributed
        if not self.run_eagerly:
            kwargs = {}
            if self._supports_reduce_retracing:
                kwargs.update({'reduce_retracing': True})
            predict_function = tf.function(predict_function, **kwargs)
        self.predict_function = predict_function

    @traceback_utils.filter_traceback
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1):
        self._assert_compile_called('fit')
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            (x, y, sample_weight), validation_data = data_adapter_utils.train_validation_split((x, y, sample_weight), validation_split=validation_split)
        if validation_data is not None:
            val_x, val_y, val_sample_weight = data_adapter_utils.unpack_x_y_sample_weight(validation_data)
        epoch_iterator = TFEpochIterator(x=x, y=y, sample_weight=sample_weight, batch_size=batch_size, steps_per_epoch=steps_per_epoch, shuffle=shuffle, class_weight=class_weight, distribute_strategy=self.distribute_strategy, steps_per_execution=self.steps_per_execution)
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0, verbose=verbose, epochs=epochs, steps=epoch_iterator.num_batches, model=self)
        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator.enumerate_epoch():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(step, self._pythonify_logs(logs))
                    if self.stop_training:
                        break
            epoch_logs = self.get_metrics_result()
            if validation_data is not None and self._should_eval(epoch, validation_freq):
                if getattr(self, '_eval_epoch_iterator', None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(x=val_x, y=val_y, sample_weight=val_sample_weight, batch_size=validation_batch_size or batch_size, distribute_strategy=self.distribute_strategy, steps_per_execution=self.steps_per_execution, steps_per_epoch=validation_steps, shuffle=False)
                val_logs = self.evaluate(x=val_x, y=val_y, sample_weight=val_sample_weight, batch_size=validation_batch_size or batch_size, steps=validation_steps, callbacks=callbacks, return_dict=True, _use_cached_eval_dataset=True)
                val_logs = {'val_' + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)
            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break
        if isinstance(self.optimizer, optimizers_module.Optimizer) and epochs > 0:
            self.optimizer.finalize_variable_values(self.trainable_weights)
        if getattr(self, '_eval_epoch_iterator', None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(self, x=None, y=None, batch_size=None, verbose='auto', sample_weight=None, steps=None, callbacks=None, return_dict=False, **kwargs):
        self._assert_compile_called('evaluate')
        use_cached_eval_dataset = kwargs.pop('_use_cached_eval_dataset', False)
        if kwargs:
            raise ValueError(f'Arguments not recognized: {kwargs}')
        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            epoch_iterator = TFEpochIterator(x=x, y=y, sample_weight=sample_weight, batch_size=batch_size, steps_per_epoch=steps, shuffle=False, distribute_strategy=self.distribute_strategy, steps_per_execution=self.steps_per_execution)
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0, verbose=verbose, epochs=1, steps=epoch_iterator.num_batches, model=self)
        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_test_batch_begin(step)
                logs = self.test_function(iterator)
                callbacks.on_test_batch_end(step, self._pythonify_logs(logs))
                if self.stop_evaluating:
                    break
        logs = self.get_metrics_result()
        callbacks.on_test_end(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(self, x, batch_size=None, verbose='auto', steps=None, callbacks=None):
        epoch_iterator = TFEpochIterator(x=x, batch_size=batch_size, steps_per_epoch=steps, shuffle=False, distribute_strategy=self.distribute_strategy, steps_per_execution=self.steps_per_execution)
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0, verbose=verbose, epochs=1, steps=epoch_iterator.num_batches, model=self)

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tf.nest.map_structure(lambda batch_output: [batch_output], batch_outputs)
            else:
                tree.map_structure_up_to(batch_outputs, lambda output, batch_output: output.append(batch_output), outputs, batch_outputs)
            return outputs

        def get_data(iterator):
            """Returns data for the next execution."""
            data = []
            for _ in range(self.steps_per_execution):
                try:
                    single_step_data = next(iterator)
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    if hasattr(data, '__len__') and len(data) > 0:
                        return data
                    else:
                        raise e
                data.append(single_step_data)
            return data
        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_predict_batch_begin(step)
                data = get_data(iterator)
                batch_outputs = self.predict_function(data)
                outputs = append_to_outputs(batch_outputs, outputs)
                callbacks.on_predict_batch_end(step, {'outputs': batch_outputs})
                if self.stop_predicting:
                    break
        callbacks.on_predict_end()
        outputs = tree.map_structure_up_to(batch_outputs, potentially_ragged_concat, outputs)
        return tf.nest.map_structure(convert_to_np_if_not_ragged, outputs)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, return_dict=False):
        self._assert_compile_called('train_on_batch')
        self.make_train_function()
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(f'Arguments `sample_weight` and `class_weight` cannot be specified at the same time. Received: sample_weight={sample_weight}, class_weight={class_weight}')
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(y, class_weight)

        def data():
            yield (x, y, sample_weight)
        logs = self.train_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(self, x, y=None, sample_weight=None, return_dict=False):
        self._assert_compile_called('test_on_batch')
        self.make_test_function()

        def data():
            yield (x, y, sample_weight)
        logs = self.test_function(data())
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tf.nest.map_structure(convert_to_np_if_not_ragged, batch_outputs)
        return batch_outputs

    @property
    def compiled_metrics(self):

        class DeprecatedCompiledMetric:

            def update_state(_, y, y_pred, sample_weight=None):
                return self._compiled_metrics_update_state(y, y_pred, sample_weight=sample_weight)
        return DeprecatedCompiledMetric()

    def _compiled_metrics_update_state(self, y, y_pred, sample_weight=None):
        warnings.warn('`model.compiled_metrics()` is deprecated. Instead, use e.g.:\n```\nfor metric in self.metrics:\n    metric.update_state(y, y_pred)\n```\n', stacklevel=2)
        for metric in self.metrics:
            if isinstance(metric, metrics_module.Mean):
                metric.update_state(y_pred, sample_weight=sample_weight)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

    def compiled_loss(self, y, y_pred, sample_weight=None, regularization_losses=None):
        warnings.warn('`model.compiled_loss()` is deprecated. Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.')
        return self.compute_loss(x=None, y=y, y_pred=y_pred, sample_weight=sample_weight)

    def loss(self, y, y_pred, sample_weight=None):
        warnings.warn('`model.loss` is deprecated. Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.')
        return self.compute_loss(x=None, y=y, y_pred=y_pred, sample_weight=sample_weight)
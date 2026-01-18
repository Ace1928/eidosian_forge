import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class _TrainingEndpoint(object):
    """A container for the training output/target and related entities.

  In the case of model with multiple outputs, there is a one-to-one mapping
  between model output (y_pred), model target (y_true), loss, metrics etc.
  By unifying these entities into one class, different entity can access
  information between each other, rather than currently access different list of
  attributes of the model.
  """

    def __init__(self, output, output_name, loss_fn, loss_weight=None, training_target=None, output_loss_metric=None, sample_weight=None, sample_weight_mode=None):
        """Initialize the _TrainingEndpoint.

    Note that the output and output_name should be stable as long as the model
    structure doesn't change. The training_target suppose to be mutable since
    the information is provided via `compile()`

    Args:
      output: the output tensor of the model.
      output_name: the unique name of the output tensor.
      loss_fn: the loss function for the output tensor.
      loss_weight: float, the weights for the loss.
      training_target: the _TrainingTarget for the model.
      output_loss_metric: the metric object for the loss function.
      sample_weight: the weights for how a sample is weighted during metric and
        loss calculation. Could be None.
      sample_weight_mode: string, 'temporal', 'samplewise' or None. The mode for
        how the sample_weight is populated.
    """
        self._output = output
        self._output_name = output_name
        self._loss_fn = loss_fn
        self._loss_weight = loss_weight
        self._training_target = training_target
        self._output_loss_metric = output_loss_metric
        self._sample_weight = sample_weight
        self._sample_weight_mode = sample_weight_mode

    @property
    def output(self):
        return self._output

    @property
    def output_name(self):
        return self._output_name

    @property
    def shape(self):
        return backend.int_shape(self.output)

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def loss_weight(self):
        return self._loss_weight

    @loss_weight.setter
    def loss_weight(self, value):
        self._loss_weight = value

    @property
    def training_target(self):
        return self._training_target

    @training_target.setter
    def training_target(self, value):
        self._training_target = value

    def create_training_target(self, target, run_eagerly=False):
        """Create training_target instance and update the self.training_target.

    Note that the input target should just be a tensor or None, and
    corresponding training target will be created based on the output and
    loss_fn.

    Args:
      target: the target tensor for the current output. Could be None.
      run_eagerly: boolean, whether the model is in run_eagerly mode.

    Raises:
      ValueError if the training_target field for the current instance has
      already been populated.
    """
        if self.has_training_target():
            raise ValueError('The training_target field for the _TrainingEndpoint instance has already been populated')
        if run_eagerly:
            self.training_target = _TrainingTarget(None, feedable=True, skip_target_weights=False)
            return
        if self.should_skip_target():
            self.training_target = _TrainingTarget(None)
        else:
            if target is not None and (not backend.is_placeholder(target)):
                feedable = False
                skip_target_weights = True
            else:
                feedable = True
                skip_target_weights = False
            if target is None:
                target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(self.loss_fn, backend.dtype(self.output))
                target = backend.placeholder(ndim=len(self.shape), name=self.output_name + '_target', sparse=backend.is_sparse(self.output), dtype=target_dtype)
            self.training_target = _TrainingTarget(target, feedable=feedable, skip_target_weights=skip_target_weights)

    @property
    def output_loss_metric(self):
        return self._output_loss_metric

    @output_loss_metric.setter
    def output_loss_metric(self, value):
        self._output_loss_metric = value

    @property
    def sample_weight(self):
        return self._sample_weight

    @sample_weight.setter
    def sample_weight(self, value):
        self._sample_weight = value

    @property
    def sample_weight_mode(self):
        return self._sample_weight_mode

    @sample_weight_mode.setter
    def sample_weight_mode(self, value):
        self._sample_weight_mode = value

    def should_skip_target(self):
        return self._loss_fn is None

    def should_skip_target_weights(self):
        return self.should_skip_target() or self.training_target is None or self.training_target.skip_target_weights

    def has_training_target(self):
        return self.training_target is not None

    def has_feedable_training_target(self):
        return not self.should_skip_target() and self.training_target is not None and self.training_target.feedable

    def loss_name(self):
        if self._loss_fn is not None:
            return self._output_name + '_loss'
        return None

    @property
    def feed_output_shape(self):
        """The output shape for the feedable target."""
        if not self.has_feedable_training_target():
            return None
        if isinstance(self.loss_fn, losses.LossFunctionWrapper) and self.loss_fn.fn == losses.sparse_categorical_crossentropy or isinstance(self.loss_fn, losses.SparseCategoricalCrossentropy):
            if backend.image_data_format() == 'channels_first':
                return (self.shape[0], 1) + self.shape[2:]
            else:
                return self.shape[:-1] + (1,)
        elif not isinstance(self.loss_fn, losses.Loss) or (isinstance(self.loss_fn, losses.LossFunctionWrapper) and getattr(losses, self.loss_fn.fn.__name__, None) is None):
            return None
        else:
            return self.shape

    def sample_weights_mismatch(self):
        """Check if the sample weight and the mode match or not."""
        return self.sample_weight_mode is not None and self.sample_weight is None or (self.sample_weight_mode is None and self.sample_weight is not None)

    def populate_sample_weight(self, sample_weight, sample_weight_mode):
        """Populate the sample weight and based on the sample weight mode."""
        if sample_weight is None and (self.should_skip_target_weights() or sample_weight_mode is None or context.executing_eagerly()):
            self._sample_weight = None
            return
        assert sample_weight_mode in ['temporal', 'samplewise']
        if sample_weight_mode == 'temporal':
            default_value = [[1.0]]
            shape = [None, None]
        else:
            default_value = [1.0]
            shape = [None]
        if sample_weight is not None:
            if not sample_weight.shape.is_compatible_with(shape):
                raise ValueError('Received sample weight with shape {}. Expected shape {}.'.format(sample_weight.shape, shape))
            self._sample_weight = sample_weight
        else:
            self._sample_weight = array_ops.placeholder_with_default(constant_op.constant(default_value, dtype=backend.floatx()), shape=shape, name=self.output_name + '_sample_weights')
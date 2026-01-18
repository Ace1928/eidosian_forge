import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
class TrackableWeightHandler(object):
    """Keras wrapper for handling tracking.Trackable object saving and restoring.

  This class handles Trackables in both V1 and V2 modes, ensuring that they can
  be saved and restored with the correct data and without adding additional ops
  on every save.

  Attributes:
    trackable: The trackable to wrap.
    num_tensors: The number of tensors that this trackable requires for saving.
  """

    def __init__(self, trackable):
        if not isinstance(trackable, tracking.Trackable):
            raise ValueError('%s is not a Trackable object.' % (trackable,))
        self._trackable = trackable
        self._distribute_strategy = distribute_lib.get_strategy()
        saveables = saveable_object_util.saveable_objects_from_trackable(trackable).values()
        if not saveables:
            self._num_tensors = 0
            self._setter = lambda weights: None
            self._getter = lambda: []
        elif len(saveables) == 1:
            saveable = list(saveables)[0]
            if ops.executing_eagerly_outside_functions():
                self._saveable = saveable
                self._num_tensors = len(self._saveable().specs)
                self._setter = lambda weights: self._saveable().restore(weights, None)
                self._getter = lambda: [spec.tensor for spec in self._saveable().specs]
            else:
                self._placeholder_tensors = []
                self._saveable = saveable()
                self._num_tensors = len(self._saveable.specs)
                for spec in self._saveable.specs:
                    tensor = spec.tensor
                    self._placeholder_tensors.append(array_ops.placeholder(tensor.dtype, tensor.shape))
                self._assign_op = self._saveable.restore(self._placeholder_tensors, None)
                self._setter = self._set_weights_v1
                self._getter = lambda: [spec.tensor for spec in self._saveable.specs]
        else:
            raise ValueError('Only Trackables with one Saveable are supported. The Trackable %s has %d Saveables.' % (trackable, len(saveables)))

    @property
    def num_tensors(self):
        return self._num_tensors

    def set_weights(self, weights):
        if len(weights) != self._num_tensors:
            raise ValueError(('Weight handler for trackable %s received the wrong number of ' + 'weights: expected %s, got %s.') % (self._trackable, self._num_tensors, len(weights)))
        self._setter(weights)

    def get_tensors(self):
        return self._getter()

    def _set_weights_v1(self, weights):
        feed_dict = {}
        for idx, tensor in enumerate(weights):
            feed_dict[self._placeholder_tensors[idx]] = tensor
        backend.get_session().run(self._assign_op, feed_dict)
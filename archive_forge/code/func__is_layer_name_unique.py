import copy
import warnings
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.module import module
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _is_layer_name_unique(self, layer):
    for ref_layer in self.layers:
        if layer.name == ref_layer.name and ref_layer is not layer:
            return False
    return True
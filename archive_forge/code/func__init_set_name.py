import copy
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.legacy_tf_layers import variable_scope_shim
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _init_set_name(self, name):
    if isinstance(name, vs.VariableScope):
        base_name = name.name
        self._name, _ = self._make_unique_name()
    else:
        base_name = name
        self._name = name
    if not name:
        self._name, base_name = self._make_unique_name()
    self._base_name = base_name
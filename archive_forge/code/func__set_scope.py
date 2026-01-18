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
def _set_scope(self, scope=None):
    if self._scope is None:
        if self._reuse:
            with vs.variable_scope(scope if scope is not None else self._base_name) as captured_scope:
                self._scope = captured_scope
        else:
            with vs.variable_scope(scope, default_name=self._base_name) as captured_scope:
                self._scope = captured_scope
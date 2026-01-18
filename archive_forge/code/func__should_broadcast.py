import copy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _should_broadcast(self, obj):
    if not nest.is_nested(obj):
        return True
    return isinstance(obj, (list, tuple)) and (not any((nest.is_nested(o) for o in obj)))
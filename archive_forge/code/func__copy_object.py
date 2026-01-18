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
def _copy_object(self, obj):
    if isinstance(obj, metrics_mod.Metric):
        return obj.__class__.from_config(obj.get_config())
    return obj
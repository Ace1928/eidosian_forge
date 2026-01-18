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
def _get_metric_objects(self, metrics, y_t, y_p):
    """Convert user-supplied metrics to `Metric` objects."""
    metrics = nest.flatten(metrics)
    return [self._get_metric_object(m, y_t, y_p) for m in metrics]
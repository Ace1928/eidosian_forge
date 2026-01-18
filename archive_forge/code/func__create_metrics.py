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
def _create_metrics(self):
    """Creates per-output loss metrics, but only for multi-output Models."""
    if len(self._output_names) == 1:
        self._per_output_metrics = [None]
    else:
        self._per_output_metrics = []
        for loss_obj, output_name in zip(self._losses, self._output_names):
            if loss_obj is None:
                self._per_output_metrics.append(None)
            else:
                self._per_output_metrics.append(metrics_mod.Mean(output_name + '_loss'))
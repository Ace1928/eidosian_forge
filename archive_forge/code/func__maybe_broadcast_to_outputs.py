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
def _maybe_broadcast_to_outputs(self, outputs, objects):
    """Determines if losses / metrics should be applied to all outputs.

    NOTE: This method should only be called for Metrics / Losses, not for
    y_true / sample_weight.

    Args:
      outputs: Model predictions.
      objects: Arbitrary nested structure (e.g. of losses or metrics)

    Returns:
      Arbitrary nested structure of objects, maybe copied to each output.

    Applies a Loss / Metric to all outputs.
    """
    if not self._should_broadcast(objects):
        return objects
    should_copy_objects = len(nest.flatten(outputs)) > 1

    def _broadcast_fn():
        if should_copy_objects:
            return nest.map_structure(self._copy_object, objects)
        return objects
    return nest.map_structure(lambda _: _broadcast_fn(), outputs)
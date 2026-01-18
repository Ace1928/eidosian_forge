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
def _conform_to_outputs(self, outputs, struct):
    """Convenience method to conform `struct` to `outputs` structure.

    Mappings performed:

    (1) Map a dict to a list of outputs, using the output names.
    (2) Fill missing keys in a dict w/ `None`s.
    (3) Map a single item to all outputs.

    Args:
      outputs: Model predictions.
      struct: Arbitrary nested structure (e.g. of labels, sample_weights,
        losses, or metrics).

    Returns:
      Mapping of `struct` to `outputs` structure.
    """
    struct = map_to_output_names(outputs, self._output_names, struct)
    struct = map_missing_dict_keys(outputs, struct)
    if not nest.is_nested(struct) and nest.is_nested(outputs):
        struct = nest.map_structure(lambda _: struct, outputs)
    return struct
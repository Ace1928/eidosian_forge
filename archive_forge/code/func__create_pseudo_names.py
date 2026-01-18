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
def _create_pseudo_names(tensors, prefix):
    """Creates pseudo {input | output} names for subclassed Models.

  Warning: this function should only be used to define default
  names for `Metics` and `SavedModel`. No other use cases should
  rely on a `Model`'s input or output names.

  Example with dict:

  `{'a': [x1, x2], 'b': x3}` becomes:
  `['a_1', 'a_2', 'b']`

  Example with list:

  `[x, y]` becomes:
  `['output_1', 'output_2']`

  Args:
    tensors: `Model`'s outputs or inputs.
    prefix: 'output_' for outputs, 'input_' for inputs.

  Returns:
    Flattened list of pseudo names.
  """

    def one_index(ele):
        if isinstance(ele, int):
            return ele + 1
        return ele
    flat_paths = list(nest.yield_flat_paths(tensors))
    flat_paths = nest.map_structure(one_index, flat_paths)
    names = []
    for path in flat_paths:
        if not path:
            name = prefix + '1'
        else:
            name = '_'.join((str(p) for p in path))
            if isinstance(path[0], int):
                name = prefix + name
        names.append(name)
    return names
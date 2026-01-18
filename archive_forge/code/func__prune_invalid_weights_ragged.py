from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
def _prune_invalid_weights_ragged(ids, weights):
    """Prune invalid weights (< 0) from the input ids and weights."""
    if weights is not None:
        is_weights_valid = math_ops.greater(weights.values, 0)
        nrows = ids.nrows()
        pruned_values = array_ops.boolean_mask_v2(ids.values, is_weights_valid)
        pruned_value_rowids = array_ops.boolean_mask_v2(ids.value_rowids(), is_weights_valid)
        ids = ragged_tensor.RaggedTensor.from_value_rowids(pruned_values, pruned_value_rowids, nrows=nrows, validate=False)
        pruned_weights_values = array_ops.boolean_mask_v2(weights.values, is_weights_valid)
        weights = ragged_tensor.RaggedTensor.from_value_rowids(pruned_weights_values, pruned_value_rowids, nrows=nrows, validate=False)
    return (ids, weights)
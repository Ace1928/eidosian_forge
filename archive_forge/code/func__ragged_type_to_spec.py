from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
def _ragged_type_to_spec(t):
    if isinstance(t, ragged_tensor.RaggedTensorType):
        return ragged_tensor.RaggedTensorSpec(None, t.dtype, t.ragged_rank - 1, t.row_splits_dtype)
    else:
        return t
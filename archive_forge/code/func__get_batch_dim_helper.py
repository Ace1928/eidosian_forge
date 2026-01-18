from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType
def _get_batch_dim_helper(v: TensorStructType) -> int:
    """Tries to find the batch dimension size of v, or None."""
    if isinstance(v, dict):
        for u in v.values():
            return _get_batch_dim_helper(u)
    elif isinstance(v, tuple):
        return _get_batch_dim_helper(v[0])
    elif isinstance(v, RepeatedValues):
        return _get_batch_dim_helper(v.values)
    else:
        B = v.shape[0]
        if hasattr(B, 'value'):
            B = B.value
        return B
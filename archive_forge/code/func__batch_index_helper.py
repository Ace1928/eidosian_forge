from typing import List
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import TensorType, TensorStructType
def _batch_index_helper(v: TensorStructType, i: int, j: int) -> TensorStructType:
    """Selects the item at the ith batch index and jth repetition."""
    if isinstance(v, dict):
        return {k: _batch_index_helper(u, i, j) for k, u in v.items()}
    elif isinstance(v, tuple):
        return tuple((_batch_index_helper(u, i, j) for u in v))
    elif isinstance(v, list):
        return _batch_index_helper(v[j], i, j)
    elif isinstance(v, RepeatedValues):
        unbatched = v.unbatch_all()
        return unbatched[i]
    else:
        return v[i, ...]
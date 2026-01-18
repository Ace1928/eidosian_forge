import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
@PublicAPI
class Std(_AggregateOnKeyBase):
    """Defines standard deviation aggregation.

    Uses Welford's online method for an accumulator-style computation of the
    standard deviation. This method was chosen due to its numerical
    stability, and it being computable in a single pass.
    This may give different (but more accurate) results than NumPy, Pandas,
    and sklearn, which use a less numerically stable two-pass algorithm.
    See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, on: Optional[str]=None, ddof: int=1, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        self._set_key_fn(on)
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'std({str(on)})'

        def merge(a: List[float], b: List[float]):
            M2_a, mean_a, count_a = a
            M2_b, mean_b, count_b = b
            delta = mean_b - mean_a
            count = count_a + count_b
            mean = (mean_a * count_a + mean_b * count_b) / count
            M2 = M2_a + M2_b + delta ** 2 * count_a * count_b / count
            return [M2, mean, count]
        null_merge = _null_wrap_merge(ignore_nulls, merge)

        def vectorized_std(block: Block) -> AggType:
            block_acc = BlockAccessor.for_block(block)
            count = block_acc.count(on)
            if count == 0 or count is None:
                return None
            sum_ = block_acc.sum(on, ignore_nulls)
            if sum_ is None:
                return None
            mean = sum_ / count
            M2 = block_acc.sum_of_squared_diffs_from_mean(on, ignore_nulls, mean)
            return [M2, mean, count]

        def finalize(a: List[float]):
            M2, mean, count = a
            if count < 2:
                return 0.0
            return math.sqrt(M2 / (count - ddof))
        super().__init__(init=_null_wrap_init(lambda k: [0, 0, 0]), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, vectorized_std, null_merge), finalize=_null_wrap_finalize(finalize), name=self._rs_name)
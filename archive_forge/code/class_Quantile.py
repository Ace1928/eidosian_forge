import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
@PublicAPI
class Quantile(_AggregateOnKeyBase):
    """Defines Quantile aggregation."""

    def __init__(self, on: Optional[str]=None, q: float=0.5, ignore_nulls: bool=True, alias_name: Optional[str]=None):
        self._set_key_fn(on)
        self._q = q
        if alias_name:
            self._rs_name = alias_name
        else:
            self._rs_name = f'quantile({str(on)})'

        def merge(a: List[int], b: List[int]):
            if isinstance(a, List) and isinstance(b, List):
                a.extend(b)
                return a
            if isinstance(a, List) and (not isinstance(b, List)):
                if b is not None and b != '':
                    a.append(b)
                return a
            if isinstance(b, List) and (not isinstance(a, List)):
                if a is not None and a != '':
                    b.append(a)
                return b
            ls = []
            if a is not None and a != '':
                ls.append(a)
            if b is not None and b != '':
                ls.append(b)
            return ls
        null_merge = _null_wrap_merge(ignore_nulls, merge)

        def block_row_ls(block: Block) -> AggType:
            block_acc = BlockAccessor.for_block(block)
            ls = []
            for row in block_acc.iter_rows(public_row_format=False):
                ls.append(row.get(on))
            return ls
        import math

        def percentile(input_values, key=lambda x: x):
            if not input_values:
                return None
            input_values = sorted(input_values)
            k = (len(input_values) - 1) * self._q
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return key(input_values[int(k)])
            d0 = key(input_values[int(f)]) * (c - k)
            d1 = key(input_values[int(c)]) * (k - f)
            return round(d0 + d1, 5)
        super().__init__(init=_null_wrap_init(lambda k: [0]), merge=null_merge, accumulate_block=_null_wrap_accumulate_block(ignore_nulls, block_row_ls, null_merge), finalize=_null_wrap_finalize(percentile), name=self._rs_name)
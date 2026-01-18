import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
class _AggregateOnKeyBase(AggregateFn):

    def _set_key_fn(self, on: str):
        self._key_fn = on

    def _validate(self, schema: Optional[Union[type, 'pa.lib.Schema']]) -> None:
        SortKey(self._key_fn).validate_schema(schema)
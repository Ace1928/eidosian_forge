from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class OrderBySpec(SpecBase):

    @no_type_check
    def __init__(self, *items: List[Any]):
        data: List[OrderItemSpec] = []
        names: Set[str] = set()
        first_ct, last_ct = (0, 0)
        for i in items:
            if not isinstance(i, OrderItemSpec):
                if isinstance(i, str):
                    i = OrderItemSpec(i)
                else:
                    i = OrderItemSpec(*list(i))
            assert_or_throw(i.name not in names, ValueError(f'{i.name} already exists'))
            names.add(i.name)
            if i.pd_na_position == 'first':
                first_ct += 1
            else:
                last_ct += 1
            data.append(i)
        pd_na_position = 'last' if last_ct == len(data) else 'first'
        super().__init__('', items=data, pd_na_position=pd_na_position)

    @property
    def items(self) -> List[OrderItemSpec]:
        return self._metadata['items']

    @property
    def keys(self) -> List[str]:
        return [x.name for x in self]

    @property
    def asc(self) -> List[bool]:
        return [x.asc for x in self]

    @property
    def pd_na_position(self) -> str:
        return self._metadata['pd_na_position']

    def __iter__(self) -> Iterator[OrderItemSpec]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
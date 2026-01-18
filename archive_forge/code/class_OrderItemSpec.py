from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class OrderItemSpec(SpecBase):

    def __init__(self, name, asc: bool=True, na_position='auto'):
        if na_position == 'auto':
            pd_na_position = 'first' if asc else 'last'
        elif na_position in ['first', 'last']:
            pd_na_position = na_position
        else:
            raise ValueError(f'{na_position} is invalid')
        super().__init__(name, asc=asc, na_position=na_position, pd_na_position=pd_na_position)

    @property
    def asc(self) -> bool:
        return self._metadata['asc']

    @property
    def na_position(self) -> str:
        return self._metadata['na_position']

    @property
    def pd_na_position(self) -> str:
        return self._metadata['pd_na_position']
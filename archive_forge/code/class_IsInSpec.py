from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class IsInSpec(PreciateSpec):

    def __init__(self, mode: str, positive: bool, *values: ArgumentSpec):
        mode = mode.lower()
        assert_or_throw(mode in ['in', 'between'], ValueError(mode))
        super().__init__(name=mode, positive=positive, values=values)

    @property
    def is_between(self) -> bool:
        return self.name.lower() == 'between'

    @property
    def values(self) -> List[ArgumentSpec]:
        return self._metadata['values']
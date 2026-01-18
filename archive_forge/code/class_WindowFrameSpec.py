from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class WindowFrameSpec(SpecBase):

    def __init__(self, frame_type: str, start: Any=None, end: Any=None):
        super().__init__(frame_type, start=start, end=end)

    @property
    def start(self) -> Any:
        return self._metadata['start']

    @property
    def end(self) -> Any:
        return self._metadata['end']
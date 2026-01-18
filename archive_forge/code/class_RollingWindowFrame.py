from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class RollingWindowFrame(WindowFrameSpec):

    def __init__(self, start: Any, end: Any):
        super().__init__('rolling', start=start, end=end)
        window = None if start is None or end is None else end - start + 1
        self._metadata['window'] = window
        assert_or_throw(self.window is None or self.window > 0, ValueError(f'invalid {start} {end}'))

    @property
    def window(self) -> int:
        return self._metadata['window']
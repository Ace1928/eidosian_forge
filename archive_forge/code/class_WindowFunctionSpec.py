from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class WindowFunctionSpec(FunctionSpec):

    def __init__(self, name: str, unique: bool, dropna: bool, window: WindowSpec):
        assert_or_throw(name in WINDOW_FUNCTIONS, ValueError(f'{name} is not an window function'))
        super().__init__(name, unique=unique, dropna=dropna, window=window)

    @property
    def window(self) -> WindowSpec:
        return self._metadata['window']

    @property
    def has_windowframe(self) -> bool:
        return not isinstance(self.window.windowframe, NoWindowFrame)

    @property
    def has_partition_by(self) -> bool:
        return len(self.window.partition_keys) > 0

    @property
    def has_order_by(self) -> bool:
        return len(self.window.order_by) > 0
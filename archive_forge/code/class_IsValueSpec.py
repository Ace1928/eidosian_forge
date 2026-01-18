from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
class IsValueSpec(PreciateSpec):

    def __init__(self, value_expr: str, positive: bool):
        super().__init__(name=value_expr.lower(), positive=positive)

    @property
    def value_expr(self) -> str:
        return self.name
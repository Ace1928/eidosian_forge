from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class TimeColumnConfig(TypedDict):
    type: Literal['time']
    format: NotRequired[str | None]
    min_value: NotRequired[str | None]
    max_value: NotRequired[str | None]
    step: NotRequired[int | float | None]
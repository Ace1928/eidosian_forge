from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class DatetimeColumnConfig(TypedDict):
    type: Literal['datetime']
    format: NotRequired[str | None]
    min_value: NotRequired[str | None]
    max_value: NotRequired[str | None]
    step: NotRequired[int | float | None]
    timezone: NotRequired[str | None]
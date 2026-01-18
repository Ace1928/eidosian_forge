from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class ProgressColumnConfig(TypedDict):
    type: Literal['progress']
    format: NotRequired[str | None]
    min_value: NotRequired[int | float | None]
    max_value: NotRequired[int | float | None]
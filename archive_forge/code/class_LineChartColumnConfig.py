from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class LineChartColumnConfig(TypedDict):
    type: Literal['line_chart']
    y_min: NotRequired[int | float | None]
    y_max: NotRequired[int | float | None]
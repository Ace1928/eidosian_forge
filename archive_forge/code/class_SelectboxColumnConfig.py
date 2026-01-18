from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class SelectboxColumnConfig(TypedDict):
    type: Literal['selectbox']
    options: NotRequired[list[str | int | float] | None]
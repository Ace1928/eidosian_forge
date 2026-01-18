from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
@dataclass(frozen=True)
class RerunData:
    """Data attached to RERUN requests. Immutable."""
    query_string: str = ''
    widget_states: WidgetStates | None = None
    page_script_hash: str = ''
    page_name: str = ''
    fragment_id_queue: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return util.repr_(self)
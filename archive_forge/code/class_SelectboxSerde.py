from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, cast
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Selectbox_pb2 import Selectbox as SelectboxProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id, save_for_app_testing
from streamlit.type_util import (
from streamlit.util import index_
@dataclass
class SelectboxSerde(Generic[T]):
    options: Sequence[T]
    index: int | None

    def serialize(self, v: object) -> int | None:
        if v is None:
            return None
        if len(self.options) == 0:
            return 0
        return index_(self.options, v)

    def deserialize(self, ui_value: int | None, widget_id: str='') -> T | None:
        idx = ui_value if ui_value is not None else self.index
        return self.options[idx] if idx is not None and len(self.options) > 0 else None
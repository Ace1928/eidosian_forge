from __future__ import annotations
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import (
from typing_extensions import TypeAlias
from streamlit import logger as _logger
from streamlit import type_util
from streamlit.deprecation_util import deprecate_func_name
from streamlit.elements.form import current_form_id
from streamlit.elements.lib.column_config_utils import (
from streamlit.elements.lib.pandas_styler_utils import marshall_styler
from streamlit.elements.utils import check_callback_rules, check_session_state_rules
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import DataFormat, DataFrameGenericAlias, Key, is_type, to_key
from streamlit.util import calc_md5
class EditingState(TypedDict, total=False):
    """
    A dictionary representing the current state of the data editor.

    Attributes
    ----------
    edited_rows : Dict[int, Dict[str, str | int | float | bool | None]]
        An hierarchical mapping of edited cells based on: row position -> column name -> value.

    added_rows : List[Dict[str, str | int | float | bool | None]]
        A list of added rows, where each row is a mapping from column name to the cell value.

    deleted_rows : List[int]
        A list of deleted rows, where each row is the numerical position of the deleted row.
    """
    edited_rows: dict[int, dict[str, str | int | float | bool | None]]
    added_rows: list[dict[str, str | int | float | bool | None]]
    deleted_rows: list[int]
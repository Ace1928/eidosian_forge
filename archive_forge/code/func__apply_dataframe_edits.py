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
def _apply_dataframe_edits(df: pd.DataFrame, data_editor_state: EditingState, dataframe_schema: DataframeSchema) -> None:
    """Apply edits to the provided dataframe (inplace).

    This includes cell edits, row additions and row deletions.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to apply the edits to.

    data_editor_state : EditingState
        The editing state of the data editor component.

    dataframe_schema: DataframeSchema
        The schema of the dataframe.
    """
    if data_editor_state.get('edited_rows'):
        _apply_cell_edits(df, data_editor_state['edited_rows'], dataframe_schema)
    if data_editor_state.get('added_rows'):
        _apply_row_additions(df, data_editor_state['added_rows'], dataframe_schema)
    if data_editor_state.get('deleted_rows'):
        _apply_row_deletions(df, data_editor_state['deleted_rows'])
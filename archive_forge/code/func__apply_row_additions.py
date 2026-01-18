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
def _apply_row_additions(df: pd.DataFrame, added_rows: list[dict[str, Any]], dataframe_schema: DataframeSchema) -> None:
    """Apply row additions to the provided dataframe (inplace).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to apply the row additions to.

    added_rows : List[Dict[str, Any]]
        A list of row additions. Each row addition is a dictionary with the
        column position as key and the new cell value as value.

    dataframe_schema: DataframeSchema
        The schema of the dataframe.
    """
    if not added_rows:
        return
    import pandas as pd
    range_index_stop = None
    range_index_step = None
    if isinstance(df.index, pd.RangeIndex):
        range_index_stop = df.index.stop
        range_index_step = df.index.step
    for added_row in added_rows:
        index_value = None
        new_row: list[Any] = [None for _ in range(df.shape[1])]
        for col_name in added_row.keys():
            value = added_row[col_name]
            if col_name == INDEX_IDENTIFIER:
                index_value = _parse_value(value, dataframe_schema[INDEX_IDENTIFIER])
            else:
                col_pos = df.columns.get_loc(col_name)
                new_row[col_pos] = _parse_value(value, dataframe_schema[col_name])
        if range_index_stop is not None:
            df.loc[range_index_stop, :] = new_row
            range_index_stop += range_index_step
        elif index_value is not None:
            df.loc[index_value, :] = new_row
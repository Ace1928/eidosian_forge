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
def _is_supported_index(df_index: pd.Index) -> bool:
    """Check if the index is supported by the data editor component.

    Parameters
    ----------

    df_index : pd.Index
        The index to check.

    Returns
    -------

    bool
        True if the index is supported, False otherwise.
    """
    import pandas as pd
    return type(df_index) in [pd.RangeIndex, pd.Index, pd.DatetimeIndex] or is_type(df_index, 'pandas.core.indexes.numeric.Int64Index') or is_type(df_index, 'pandas.core.indexes.numeric.Float64Index') or is_type(df_index, 'pandas.core.indexes.numeric.UInt64Index')
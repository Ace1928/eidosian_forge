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
def _check_type_compatibilities(data_df: pd.DataFrame, columns_config: ColumnConfigMapping, dataframe_schema: DataframeSchema):
    """Check column type to data type compatibility.

    Iterates the index and all columns of the dataframe to check if
    the configured column types are compatible with the underlying data types.

    Parameters
    ----------
    data_df : pd.DataFrame
        The dataframe to check the type compatibilities for.

    columns_config : ColumnConfigMapping
        A mapping of column to column configurations.

    dataframe_schema : DataframeSchema
        The schema of the dataframe.

    Raises
    ------
    StreamlitAPIException
        If a configured column type is editable and not compatible with the
        underlying data type.
    """
    indices = [(INDEX_IDENTIFIER, data_df.index)]
    for column in indices + list(data_df.items()):
        column_name, _ = column
        column_data_kind = dataframe_schema[column_name]
        if column_name in columns_config:
            column_config = columns_config[column_name]
            if column_config.get('disabled') is True:
                continue
            type_config = column_config.get('type_config')
            if type_config is None:
                continue
            configured_column_type = type_config.get('type')
            if configured_column_type is None:
                continue
            if is_type_compatible(configured_column_type, column_data_kind) is False:
                raise StreamlitAPIException(f'The configured column type `{configured_column_type}` for column `{column_name}` is not compatible for editing the underlying data type `{column_data_kind}`.\n\nYou have following options to fix this: 1) choose a compatible type 2) disable the column 3) convert the column into a compatible data type.')
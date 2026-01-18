from __future__ import annotations
import contextlib
import copy
import math
import re
import types
from enum import Enum, EnumMeta, auto
from typing import (
from typing_extensions import TypeAlias, TypeGuard
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
from streamlit.errors import StreamlitAPIException
def _maybe_truncate_table(table: pa.Table, truncated_rows: int | None=None) -> pa.Table:
    """Experimental feature to automatically truncate tables that
    are larger than the maximum allowed message size. It needs to be enabled
    via the server.enableArrowTruncation config option.

    Parameters
    ----------
    table : pyarrow.Table
        A table to truncate.

    truncated_rows : int or None
        The number of rows that have been truncated so far. This is used by
        the recursion logic to keep track of the total number of truncated
        rows.

    """
    if config.get_option('server.enableArrowTruncation'):
        max_message_size = int(config.get_option('server.maxMessageSize') * 1000000.0)
        table_size = int(table.nbytes + 1 * 1000000.0)
        table_rows = table.num_rows
        if table_rows > 1 and table_size > max_message_size:
            targeted_rows = math.ceil(table_rows * (max_message_size / table_size))
            targeted_rows = math.floor(max(min(targeted_rows - math.floor((table_rows - targeted_rows) * 0.05), table_rows - table_rows * 0.01, table_rows - 5), 1))
            sliced_table = table.slice(0, targeted_rows)
            return _maybe_truncate_table(sliced_table, (truncated_rows or 0) + (table_rows - targeted_rows))
        if truncated_rows:
            displayed_rows = string_util.simplify_number(table.num_rows)
            total_rows = string_util.simplify_number(table.num_rows + truncated_rows)
            if displayed_rows == total_rows:
                displayed_rows = str(table.num_rows)
                total_rows = str(table.num_rows + truncated_rows)
            st.caption(f'⚠️ Showing {displayed_rows} out of {total_rows} rows due to data size limitations.')
    return table
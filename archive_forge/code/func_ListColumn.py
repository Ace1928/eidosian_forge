from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.ListColumn')
def ListColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None):
    """Configure a list column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for list-like values. List columns are not editable
    at the moment. This command needs to be used in the ``column_config`` parameter of
    ``st.dataframe`` or ``st.data_editor``.

    Parameters
    ----------

    label: str or None
        The label shown at the top of the column. If None (default),
        the column name is used.

    width: "small", "medium", "large", or None
        The display width of the column. Can be one of "small", "medium", or "large".
        If None (default), the column will be sized to fit the cell contents.

    help: str or None
        An optional tooltip that gets displayed when hovering over the column label.

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "sales": [
    >>>             [0, 4, 26, 80, 100, 40],
    >>>             [80, 20, 80, 35, 40, 100],
    >>>             [10, 20, 80, 80, 70, 0],
    >>>             [10, 100, 20, 100, 30, 100],
    >>>         ],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "sales": st.column_config.ListColumn(
    >>>             "Sales (last 6 months)",
    >>>             help="The sales volume in the last 6 months",
    >>>             width="medium",
    >>>         ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-list-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, type_config=ListColumnConfig(type='list'))
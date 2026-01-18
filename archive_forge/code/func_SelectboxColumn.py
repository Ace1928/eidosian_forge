from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.SelectboxColumn')
def SelectboxColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: str | int | float | None=None, options: Iterable[str | int | float] | None=None) -> ColumnConfig:
    """Configure a selectbox column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for Pandas categorical values. This command needs to
    be used in the ``column_config`` parameter of ``st.dataframe`` or ``st.data_editor``.
    When used with ``st.data_editor``, editing will be enabled with a selectbox widget.

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

    disabled: bool or None
        Whether editing should be disabled for this column. Defaults to False.

    required: bool or None
        Whether edited cells in the column need to have a value. If True, an edited cell
        can only be submitted if it has a value other than None. Defaults to False.

    default: str, int, float, bool, or None
        Specifies the default value in this column when a new row is added by the user.

    options: Iterable of str or None
        The options that can be selected during editing. If None (default), this will be
        inferred from the underlying dataframe column if its dtype is "category"
        (`see Pandas docs on categorical data <https://pandas.pydata.org/docs/user_guide/categorical.html>`_).

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "category": [
    >>>             "📊 Data Exploration",
    >>>             "📈 Data Visualization",
    >>>             "🤖 LLM",
    >>>             "📊 Data Exploration",
    >>>         ],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "category": st.column_config.SelectboxColumn(
    >>>             "App Category",
    >>>             help="The category of the app",
    >>>             width="medium",
    >>>             options=[
    >>>                 "📊 Data Exploration",
    >>>                 "📈 Data Visualization",
    >>>                 "🤖 LLM",
    >>>             ],
    >>>             required=True,
    >>>         )
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-selectbox-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=default, type_config=SelectboxColumnConfig(type='selectbox', options=list(options) if options is not None else None))
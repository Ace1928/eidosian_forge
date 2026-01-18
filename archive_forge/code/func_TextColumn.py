from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.TextColumn')
def TextColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, disabled: bool | None=None, required: bool | None=None, default: str | None=None, max_chars: int | None=None, validate: str | None=None) -> ColumnConfig:
    """Configure a text column in ``st.dataframe`` or ``st.data_editor``.

    This is the default column type for string values. This command needs to be used in the
    ``column_config`` parameter of ``st.dataframe`` or ``st.data_editor``. When used with
    ``st.data_editor``, editing will be enabled with a text input widget.

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

    default: str or None
        Specifies the default value in this column when a new row is added by the user.

    max_chars: int or None
        The maximum number of characters that can be entered. If None (default),
        there will be no maximum.

    validate: str or None
        A regular expression (JS flavor, e.g. ``"^[a-z]+$"``) that edited values are validated against.
        If the input is invalid, it will not be submitted.

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "widgets": ["st.selectbox", "st.number_input", "st.text_area", "st.button"],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "widgets": st.column_config.TextColumn(
    >>>             "Widgets",
    >>>             help="Streamlit **widget** commands 🎈",
    >>>             default="st.",
    >>>             max_chars=50,
    >>>             validate="^st\\.[a-z_]+$",
    >>>         )
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-text-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, disabled=disabled, required=required, default=default, type_config=TextColumnConfig(type='text', max_chars=max_chars, validate=validate))
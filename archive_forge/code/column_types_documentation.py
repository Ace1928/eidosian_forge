from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
Configure a progress column in ``st.dataframe`` or ``st.data_editor``.

    Cells need to contain a number. Progress columns are not editable at the moment.
    This command needs to be used in the ``column_config`` parameter of ``st.dataframe``
    or ``st.data_editor``.

    Parameters
    ----------

    label : str or None
        The label shown at the top of the column. If None (default),
        the column name is used.

    width : "small", "medium", "large", or None
        The display width of the column. Can be one of "small", "medium", or "large".
        If None (default), the column will be sized to fit the cell contents.

    help : str or None
        An optional tooltip that gets displayed when hovering over the column label.

    format : str or None
        A printf-style format string controlling how numbers are displayed.
        Valid formatters: %d %e %f %g %i %u. You can also add prefixes and suffixes,
        e.g. ``"$ %.2f"`` to show a dollar prefix.

    min_value : int, float, or None
        The minimum value of the progress bar.
        If None (default), will be 0.

    max_value : int, float, or None
        The minimum value of the progress bar. If None (default), will be 100 for
        integer values and 1 for float values.

    Examples
    --------

    >>> import pandas as pd
    >>> import streamlit as st
    >>>
    >>> data_df = pd.DataFrame(
    >>>     {
    >>>         "sales": [200, 550, 1000, 80],
    >>>     }
    >>> )
    >>>
    >>> st.data_editor(
    >>>     data_df,
    >>>     column_config={
    >>>         "sales": st.column_config.ProgressColumn(
    >>>             "Sales volume",
    >>>             help="The sales volume in USD",
    >>>             format="$%f",
    >>>             min_value=0,
    >>>             max_value=1000,
    >>>         ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-progress-column.streamlit.app/
        height: 300px
    
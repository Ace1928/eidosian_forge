from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
@gather_metrics('column_config.LineChartColumn')
def LineChartColumn(label: str | None=None, *, width: ColumnWidth | None=None, help: str | None=None, y_min: int | float | None=None, y_max: int | float | None=None) -> ColumnConfig:
    """Configure a line chart column in ``st.dataframe`` or ``st.data_editor``.

    Cells need to contain a list of numbers. Chart columns are not editable
    at the moment. This command needs to be used in the ``column_config`` parameter
    of ``st.dataframe`` or ``st.data_editor``.

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

    y_min: int, float, or None
        The minimum value on the y-axis for all cells in the column.
        If None (default), every cell will use the minimum of its data.

    y_max: int, float, or None
        The maximum value on the y-axis for all cells in the column. If None (default),
        every cell will use the maximum of its data.

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
    >>>         "sales": st.column_config.LineChartColumn(
    >>>             "Sales (last 6 months)",
    >>>             width="medium",
    >>>             help="The sales volume in the last 6 months",
    >>>             y_min=0,
    >>>             y_max=100,
    >>>          ),
    >>>     },
    >>>     hide_index=True,
    >>> )

    .. output::
        https://doc-linechart-column.streamlit.app/
        height: 300px
    """
    return ColumnConfig(label=label, width=width, help=help, type_config=LineChartColumnConfig(type='line_chart', y_min=y_min, y_max=y_max))
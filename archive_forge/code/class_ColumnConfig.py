from __future__ import annotations
import datetime
from typing import Iterable, Literal, TypedDict
from typing_extensions import NotRequired, TypeAlias
from streamlit.runtime.metrics_util import gather_metrics
class ColumnConfig(TypedDict, total=False):
    """Configuration options for columns in ``st.dataframe`` and ``st.data_editor``.

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

    default: str, bool, int, float, or None
        Specifies the default value in this column when a new row is added by the user.

    hidden: bool or None
        Whether to hide the column. Defaults to False.

    type_config: dict or str or None
        Configure a column type and type specific options.
    """
    label: str | None
    width: ColumnWidth | None
    help: str | None
    hidden: bool | None
    disabled: bool | None
    required: bool | None
    default: str | bool | int | float | None
    alignment: Literal['left', 'center', 'right'] | None
    type_config: NumberColumnConfig | TextColumnConfig | CheckboxColumnConfig | SelectboxColumnConfig | LinkColumnConfig | ListColumnConfig | DatetimeColumnConfig | DateColumnConfig | TimeColumnConfig | ProgressColumnConfig | LineChartColumnConfig | BarChartColumnConfig | AreaChartColumnConfig | ImageColumnConfig | None
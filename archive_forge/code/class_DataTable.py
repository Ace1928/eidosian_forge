from __future__ import annotations
import logging # isort:skip
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.singletons import Intrinsic
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
class DataTable(TableWidget):
    """ Two-dimensional grid for visualization and editing large amounts
    of data.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    autosize_mode = Enum(AutosizeMode, default='force_fit', help='\n    Describes the column autosizing mode with one of the following options:\n\n    ``"fit_columns"``\n        Compute column widths based on cell contents but ensure the\n        table fits into the available viewport. This results in no\n        horizontal scrollbar showing up, but data can get unreadable\n        if there is not enough space available.\n\n    ``"fit_viewport"``\n        Adjust the viewport size after computing columns widths based\n        on cell contents.\n\n    ``"force_fit"``\n        Fit columns into available space dividing the table width across\n        the columns equally (equivalent to `fit_columns=True`).\n        This results in no horizontal scrollbar showing up, but data\n        can get unreadable if there is not enough space available.\n\n    ``"none"``\n        Do not automatically compute column widths.\n    ')
    auto_edit = Bool(False, help='\n    When enabled editing mode is enabled after a single click on a\n    table cell.\n    ')
    columns = List(Instance(TableColumn), help='\n    The list of child column widgets.\n    ')
    fit_columns = Nullable(Bool, help="\n    **This is a legacy parameter.** For new development, use the\n    ``autosize_mode`` parameter.\n\n    Whether columns should be fit to the available width. This results in\n    no horizontal scrollbar showing up, but data can get unreadable if there\n    is not enough space available. If set to True, each column's width is\n    understood as maximum width.\n    ")
    frozen_columns = Nullable(Int, help='\n    Integer indicating the number of columns to freeze. If set the first N\n    columns will be frozen which prevents them from scrolling out of frame.\n    ')
    frozen_rows = Nullable(Int, help='\n    Integer indicating the number of rows to freeze. If set the first N\n    rows will be frozen which prevents them from scrolling out of frame,\n    if set to a negative value last N rows will be frozen.\n    ')
    sortable = Bool(True, help="\n    Allows to sort table's contents. By default natural order is preserved.\n    To sort a column, click on it's header. Clicking one more time changes\n    sort direction. Use Ctrl + click to return to natural order. Use\n    Shift + click to sort multiple columns simultaneously.\n    ")
    reorderable = Bool(True, help="\n    Allows the reordering of a table's columns. To reorder a column,\n    click and drag a table's header to the desired location in the table.\n    The columns on either side will remain in their previous order.\n    ")
    editable = Bool(False, help="\n    Allows to edit table's contents. Needs cell editors to be configured on\n    columns that are required to be editable.\n    ")
    selectable = Either(Bool(True), Enum('checkbox'), help="\n    Whether a table's rows can be selected or not. Using ``checkbox`` is\n    equivalent to True, but makes selection visible through a checkbox\n    for each row, instead of highlighting rows. Multiple selection is\n    allowed and can be achieved by either clicking multiple checkboxes (if\n    enabled) or using Shift + click on rows.\n    ")
    index_position = Nullable(Int, default=0, help='\n    Where among the list of columns to insert a column displaying the row\n    index. Negative indices are supported, and specify an index position\n    from the end of the list of columns (i.e. standard Python behaviour).\n\n    To prevent the index column from being added, set to None.\n\n    If the absolute value of index_position is larger than the length of\n    the columns, then the index will appear at the beginning or end, depending\n    on the sign.\n    ')
    index_header = String('#', help='\n    The column header to display for the index column, if it is present.\n    ')
    index_width = Int(40, help='\n    The width of the index column, if present.\n    ')
    scroll_to_selection = Bool(True, help="\n    Whenever a selection is made on the data source, scroll the selected\n    rows into the table's viewport if none of the selected rows are already\n    in the viewport.\n    ")
    header_row = Bool(True, help='\n    Whether to show a header row with column names at the top of the table.\n    ')
    width = Override(default=600)
    height = Override(default=400)
    row_height = Int(25, help='\n    The height of each row in pixels.\n    ')

    @staticmethod
    def from_data(data, columns=None, formatters={}, **kwargs) -> DataTable:
        """ Create a simple table from a pandas dataframe, dictionary or ColumnDataSource.

        Args:
            data (DataFrame or dict or ColumnDataSource) :
                The data to create the table from. If the data is a dataframe
                or dictionary, a ColumnDataSource will be created from it.

            columns (list, optional) :
                A list of column names to use from the input data.
                If None, use all columns. (default: None)

            formatters (dict, optional) :
                A mapping of column names and corresponding Formatters to
                apply to each column. (default: None)

        Keyword arguments:
            Any additional keyword arguments will be passed to DataTable.

        Returns:
            DataTable

        Raises:
            ValueError
                If the provided data is not a ColumnDataSource
                or a data source that a ColumnDataSource can be created from.

        """
        if isinstance(data, ColumnDataSource):
            source = data.clone()
        else:
            try:
                source = ColumnDataSource(data)
            except ValueError as e:
                raise ValueError('Expected a ColumnDataSource or something a ColumnDataSource can be created from like a dict or a DataFrame') from e
        if columns is not None:
            source.data = {col: source.data[col] for col in columns}
        table_columns = []
        for c in source.data.keys():
            formatter = formatters.get(c, Intrinsic)
            table_columns.append(TableColumn(field=c, title=c, formatter=formatter))
        return DataTable(source=source, columns=table_columns, index_position=None, **kwargs)
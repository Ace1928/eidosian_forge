from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.models.widgets.tables import TableColumn
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class TableEditEvent(ModelEvent):
    event_name = 'table-edit'

    def __init__(self, model, column, row, pre=False, value=None, old=None):
        self.column = column
        self.row = row
        self.value = value
        self.old = old
        self.pre = pre
        super().__init__(model=model)

    def __repr__(self):
        return f'{type(self).__name__}(column={self.column}, row={self.row}, value={self.value}, old={self.old})'
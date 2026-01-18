from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import ColumnDataSource, LayoutDOM
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
class VegaEvent(ModelEvent):
    event_name = 'vega_event'

    def __init__(self, model, data=None):
        self.data = data
        super().__init__(model=model)
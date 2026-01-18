from bokeh.core.properties import (
from bokeh.events import ModelEvent
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
from .layout import HTMLBox
class JSONEditEvent(ModelEvent):
    event_name = 'json_edit'

    def __init__(self, model, data=None):
        self.data = data
        super().__init__(model=model)
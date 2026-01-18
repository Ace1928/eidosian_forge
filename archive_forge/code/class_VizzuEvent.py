from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import LayoutDOM
from bokeh.models.sources import DataSource
from ..config import config
from ..util import classproperty
class VizzuEvent(ModelEvent):
    event_name = 'vizzu_event'

    def __init__(self, model, data=None):
        self.data = data
        super().__init__(model=model)
import difflib
import re
from collections import defaultdict
from html.parser import HTMLParser
import bokeh.core.properties as bp
from bokeh.events import ModelEvent
from bokeh.model import DataModel
from bokeh.models import LayoutDOM
from .layout import HTMLBox
class DOMEvent(ModelEvent):
    event_name = 'dom_event'

    def __init__(self, model, node=None, data=None):
        self.data = data
        self.node = node
        super().__init__(model=model)
import difflib
import re
from collections import defaultdict
from html.parser import HTMLParser
import bokeh.core.properties as bp
from bokeh.events import ModelEvent
from bokeh.model import DataModel
from bokeh.models import LayoutDOM
from .layout import HTMLBox
class ReactiveHTML(HTMLBox):
    attrs = bp.Dict(bp.String, bp.List(bp.Tuple(bp.String, bp.List(bp.String), bp.String)))
    callbacks = bp.Dict(bp.String, bp.List(bp.Tuple(bp.String, bp.String)))
    children = bp.Dict(bp.String, bp.Either(bp.List(bp.Either(bp.Instance(LayoutDOM), bp.String)), bp.String))
    data = bp.Instance(DataModel)
    events = bp.Dict(bp.String, bp.Dict(bp.String, bp.Bool))
    event_params = bp.List(bp.String)
    html = bp.String()
    looped = bp.List(bp.String)
    nodes = bp.List(bp.String)
    scripts = bp.Dict(bp.String, bp.List(bp.String))

    def __init__(self, **props):
        if 'attrs' not in props and 'html' in props:
            props['attrs'] = find_attrs(props['html'])
        super().__init__(**props)
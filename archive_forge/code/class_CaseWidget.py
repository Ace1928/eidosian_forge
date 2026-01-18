from io import StringIO
from html.parser import HTMLParser
import json
import os
import re
import tempfile
import shutil
import traitlets
from ..widgets import IntSlider, IntText, Text, Widget, jslink, HBox, widget_serialization, widget as widget_module
from ..embed import embed_data, embed_snippet, embed_minimal_html, dependency_state
class CaseWidget(Widget):
    """Widget to test dependency traversal"""
    a = traitlets.Instance(Widget, allow_none=True).tag(sync=True, **widget_serialization)
    b = traitlets.Instance(Widget, allow_none=True).tag(sync=True, **widget_serialization)
    _model_name = traitlets.Unicode('CaseWidgetModel').tag(sync=True)
    other = traitlets.Dict().tag(sync=True, **widget_serialization)
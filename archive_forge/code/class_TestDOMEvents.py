from __future__ import annotations
import asyncio
import unittest.mock
from functools import partial
from typing import ClassVar, Mapping
import bokeh.core.properties as bp
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from bokeh.models import Div
from panel.depends import bind, depends
from panel.layout import Tabs, WidgetBox
from panel.pane import Markdown
from panel.reactive import Reactive, ReactiveHTML
from panel.viewable import Viewable
from panel.widgets import (
class TestDOMEvents(ReactiveHTML):
    int = param.Integer(default=3, doc='An integer')
    float = param.Number(default=3.14, doc='A float')
    _template = '<div id="div" width=${int}></div>'
    _dom_events = {'div': ['change']}
from a list that contains the shared attribute. This is required for the
import abc
import operator
import sys
from functools import partial
from packaging.version import Version
from types import FunctionType, MethodType
import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.layout import Column, Row, VSpacer, HSpacer
from panel.util import get_method_owner, full_groupby
from panel.widgets.base import Widget
from .converter import HoloViewsConverter
from .util import (
def _layout_bk3(self, **kwargs):
    widget_box = self.widgets()
    panel = self.output()
    loc = self._loc
    center = self._center
    alignments = {'left': (Row, ('start', 'center'), True), 'right': (Row, ('end', 'center'), False), 'top': (Column, ('center', 'start'), True), 'bottom': (Column, ('center', 'end'), False), 'top_left': (Column, 'start', True), 'top_right': (Column, ('end', 'start'), True), 'bottom_left': (Column, ('start', 'end'), False), 'bottom_right': (Column, 'end', False), 'left_top': (Row, 'start', True), 'left_bottom': (Row, ('start', 'end'), True), 'right_top': (Row, ('end', 'start'), False), 'right_bottom': (Row, 'end', False)}
    layout, align, widget_first = alignments[loc]
    widget_box.align = align
    if not len(widget_box):
        if center:
            components = [HSpacer(), panel, HSpacer()]
        else:
            components = [panel]
        return Row(*components, **kwargs)
    items = (widget_box, panel) if widget_first else (panel, widget_box)
    sizing_mode = kwargs.get('sizing_mode')
    if not center:
        if layout is Row:
            components = list(items)
        else:
            components = [layout(*items, sizing_mode=sizing_mode)]
    elif layout is Column:
        components = [HSpacer(), layout(*items, sizing_mode=sizing_mode), HSpacer()]
    elif loc.startswith('left'):
        components = [widget_box, HSpacer(), panel, HSpacer()]
    else:
        components = [HSpacer(), panel, HSpacer(), widget_box]
    return Row(*components, **kwargs)
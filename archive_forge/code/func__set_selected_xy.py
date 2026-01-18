from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@observe('selected')
def _set_selected_xy(self, change):
    value = self.selected
    if self.selected is None or len(self.selected) == 0:
        if not (self.selected_x is None or len(self.selected_x) == 0 or self.selected_y is None or (len(self.selected_y) == 0)):
            self.selected_x = None
            self.selected_y = None
    else:
        (x0, y0), (x1, y1) = value
        x = [x0, x1]
        y = [y0, y1]
        with self.hold_sync():
            if not _array_equal(self.selected_x, x):
                self.selected_x = x
            if not _array_equal(self.selected_y, y):
                self.selected_y = y
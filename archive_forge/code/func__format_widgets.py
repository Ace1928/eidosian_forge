from __future__ import division
import math
import os
import signal
import sys
import time
from .compat import *  # for: any, next
from . import widgets
def _format_widgets(self):
    result = []
    expanding = []
    width = self.term_width
    for index, widget in enumerate(self.widgets):
        if isinstance(widget, widgets.WidgetHFill):
            result.append(widget)
            expanding.insert(0, index)
        else:
            widget = widgets.format_updatable(widget, self)
            result.append(widget)
            width -= len(widget)
    count = len(expanding)
    while count:
        portion = max(int(math.ceil(width * 1.0 / count)), 0)
        index = expanding.pop()
        count -= 1
        widget = result[index].update(self, portion)
        width -= len(widget)
        result[index] = widget
    return result
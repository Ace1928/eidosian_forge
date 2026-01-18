import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
def _set_cursor_cbk(self, event):
    if not event or not self.canvas:
        return
    if self._current_tool and getattr(event, 'inaxes', None) and event.inaxes.get_navigate():
        if self._last_cursor != self._current_tool.cursor:
            self.canvas.set_cursor(self._current_tool.cursor)
            self._last_cursor = self._current_tool.cursor
    elif self._last_cursor != self._default_cursor:
        self.canvas.set_cursor(self._default_cursor)
        self._last_cursor = self._default_cursor
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
def _tool_trigger_cbk(self, event):
    if event.tool.toggled:
        self._current_tool = event.tool
    else:
        self._current_tool = None
    self._set_cursor_cbk(event.canvasevent)
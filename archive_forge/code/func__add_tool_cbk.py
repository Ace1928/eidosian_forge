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
def _add_tool_cbk(self, event):
    """Process every newly added tool."""
    if event.tool is self:
        return
    self._add_tool(event.tool)
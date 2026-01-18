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
def _add_tool(self, tool):
    """Set the cursor when the tool is triggered."""
    if getattr(tool, 'cursor', None) is not None:
        self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_trigger_cbk)
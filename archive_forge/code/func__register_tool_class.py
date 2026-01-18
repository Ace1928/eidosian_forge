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
def _register_tool_class(canvas_cls, tool_cls=None):
    """Decorator registering *tool_cls* as a tool class for *canvas_cls*."""
    if tool_cls is None:
        return functools.partial(_register_tool_class, canvas_cls)
    _tool_registry.add((canvas_cls, tool_cls))
    return tool_cls
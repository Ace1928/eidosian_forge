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
def _find_tool_class(canvas_cls, tool_cls):
    """Find a subclass of *tool_cls* registered for *canvas_cls*."""
    for canvas_parent in canvas_cls.__mro__:
        for tool_child in _api.recursive_subclasses(tool_cls):
            if (canvas_parent, tool_child) in _tool_registry:
                return tool_child
    return tool_cls
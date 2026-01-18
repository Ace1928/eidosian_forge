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
class ToolBack(ViewsPositionsBase):
    """Move back up the view limits stack."""
    description = 'Back to previous view'
    image = 'back'
    default_keymap = property(lambda self: mpl.rcParams['keymap.back'])
    _on_trigger = 'back'
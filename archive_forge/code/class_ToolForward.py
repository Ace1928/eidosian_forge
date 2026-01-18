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
class ToolForward(ViewsPositionsBase):
    """Move forward in the view lim stack."""
    description = 'Forward to next view'
    image = 'forward'
    default_keymap = property(lambda self: mpl.rcParams['keymap.forward'])
    _on_trigger = 'forward'
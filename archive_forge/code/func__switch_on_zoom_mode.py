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
def _switch_on_zoom_mode(self, event):
    self._zoom_mode = event.key
    self._mouse_move(event)
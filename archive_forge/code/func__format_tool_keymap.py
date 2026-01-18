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
def _format_tool_keymap(self, name):
    keymaps = self.toolmanager.get_tool_keymap(name)
    return ', '.join((self.format_shortcut(keymap) for keymap in keymaps))
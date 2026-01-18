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
class ToolCopyToClipboardBase(ToolBase):
    """Tool to copy the figure to the clipboard."""
    description = 'Copy the canvas figure to clipboard'
    default_keymap = property(lambda self: mpl.rcParams['keymap.copy'])

    def trigger(self, *args, **kwargs):
        message = 'Copy tool is not available'
        self.toolmanager.message_event(message, self)
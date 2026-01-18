import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def _on_leave(self, event):
    """Mouse has left the window."""
    event.Skip()
    LocationEvent('figure_leave_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(), guiEvent=event)._process()
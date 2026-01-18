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
@functools.lru_cache(1)
def _create_wxapp():
    wxapp = wx.App(False)
    wxapp.SetExitOnFrameDelete(True)
    cbook._setup_new_guiapp()
    return wxapp
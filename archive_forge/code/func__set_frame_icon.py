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
def _set_frame_icon(frame):
    bundle = wx.IconBundle()
    for image in ('matplotlib.png', 'matplotlib_large.png'):
        icon = wx.Icon(_load_bitmap(image))
        if not icon.IsOk():
            return
        bundle.AddIcon(icon)
    frame.SetIcons(bundle)
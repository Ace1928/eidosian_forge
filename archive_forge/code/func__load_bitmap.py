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
def _load_bitmap(filename):
    """
    Load a wx.Bitmap from a file in the "images" directory of the Matplotlib
    data.
    """
    return wx.Bitmap(str(cbook._get_data_path('images', filename)))
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
@_Backend.export
class _BackendWx(_Backend):
    FigureCanvas = FigureCanvasWx
    FigureManager = FigureManagerWx
    mainloop = FigureManagerWx.start_main_loop
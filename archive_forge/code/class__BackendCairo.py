import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
@_Backend.export
class _BackendCairo(_Backend):
    backend_version = cairo.version
    FigureCanvas = FigureCanvasCairo
    FigureManager = FigureManagerBase
from __future__ import annotations
from contextlib import AbstractContextManager, ExitStack
from enum import Enum
import functools
import importlib
import inspect
import logging
import re
import sys
import threading
import time
from typing import cast, overload
from cycler import cycler
import matplotlib
import matplotlib.colorbar
import matplotlib.image
from matplotlib import _api
from matplotlib import (  # Re-exported for typing.
from matplotlib import _pylab_helpers, interactive
from matplotlib import cbook
from matplotlib import _docstring
from matplotlib.backend_bases import (
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.artist import Artist
from matplotlib.axes import Axes, Subplot  # type: ignore
from matplotlib.projections import PolarAxes  # type: ignore
from matplotlib import mlab  # for detrend_none, window_hanning
from matplotlib.scale import get_scale_names
from matplotlib.cm import _colormaps
from matplotlib.cm import register_cmap  # type: ignore
from matplotlib.colors import _color_sequences
import numpy as np
from typing import TYPE_CHECKING, cast
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D, AxLine
from matplotlib.text import Text, Annotation
from matplotlib.patches import Polygon, Rectangle, Circle, Arrow
from matplotlib.widgets import Button, Slider, Widget
from .ticker import (
def _warn_if_gui_out_of_main_thread() -> None:
    warn = False
    canvas_class = cast(type[FigureCanvasBase], _get_backend_mod().FigureCanvas)
    if canvas_class.required_interactive_framework:
        if hasattr(threading, 'get_native_id'):
            if threading.get_native_id() != threading.main_thread().native_id:
                warn = True
        elif threading.current_thread() is not threading.main_thread():
            warn = True
    if warn:
        _api.warn_external('Starting a Matplotlib GUI outside of the main thread will likely fail.')
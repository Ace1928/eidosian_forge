from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
def _contour_labeler_event_handler(cs, inline, inline_spacing, event):
    canvas = cs.axes.figure.canvas
    is_button = event.name == 'button_press_event'
    is_key = event.name == 'key_press_event'
    if is_button and event.button == MouseButton.MIDDLE or (is_key and event.key in ['escape', 'enter']):
        canvas.stop_event_loop()
    elif is_button and event.button == MouseButton.RIGHT or (is_key and event.key in ['backspace', 'delete']):
        if not inline:
            cs.pop_label()
            canvas.draw()
    elif is_button and event.button == MouseButton.LEFT or (is_key and event.key is not None):
        if cs.axes.contains(event)[0]:
            cs.add_label_near(event.x, event.y, transform=False, inline=inline, inline_spacing=inline_spacing)
            canvas.draw()
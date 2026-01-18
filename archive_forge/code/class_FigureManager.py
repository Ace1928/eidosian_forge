import sys
import types
from warnings import warn
import io
import json
from base64 import b64encode
import matplotlib
import numpy as np
from IPython import get_ipython
from IPython import version_info as ipython_version_info
from IPython.display import HTML, display
from ipython_genutils.py3compat import string_types
from ipywidgets import DOMWidget, widget_serialization
from matplotlib import is_interactive, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import NavigationToolbar2, _Backend, cursors
from matplotlib.backends.backend_webagg_core import (
from PIL import Image
from traitlets import (
from ._version import js_semver
class FigureManager(FigureManagerWebAgg):
    if matplotlib.__version__ < '3.6':
        ToolbarCls = Toolbar

    def __init__(self, canvas, num):
        FigureManagerWebAgg.__init__(self, canvas, num)
        self.web_sockets = [self.canvas]
        self.toolbar = Toolbar(self.canvas)

    def show(self):
        if self.canvas._closed:
            self.canvas._closed = False
            display(self.canvas)
        else:
            self.canvas.draw_idle()

    def destroy(self):
        self.canvas.close()
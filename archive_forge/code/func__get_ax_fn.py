from a list that contains the shared attribute. This is required for the
import abc
import operator
import sys
from functools import partial
from packaging.version import Version
from types import FunctionType, MethodType
import holoviews as hv
import pandas as pd
import panel as pn
import param
from panel.layout import Column, Row, VSpacer, HSpacer
from panel.util import get_method_owner, full_groupby
from panel.widgets.base import Widget
from .converter import HoloViewsConverter
from .util import (
@staticmethod
def _get_ax_fn():

    @pn.depends()
    def get_ax():
        from matplotlib.backends.backend_agg import FigureCanvas
        from matplotlib.pyplot import Figure
        Interactive._fig = fig = Figure()
        FigureCanvas(fig)
        return fig.subplots()
    return get_ax
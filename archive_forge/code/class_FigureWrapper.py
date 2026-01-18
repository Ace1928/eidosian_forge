from __future__ import annotations
import asyncio
import inspect
import itertools
import json
import os
import sys
import textwrap
import types
from collections import defaultdict, namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import (
import param
from param.parameterized import (
from param.reactive import rx
from .config import config
from .io import state
from .layout import (
from .pane import DataFrame as DataFramePane
from .pane.base import PaneBase, ReplacementPane
from .reactive import Reactive
from .util import (
from .util.checks import is_dataframe, is_mpl_axes, is_series
from .viewable import Layoutable, Viewable
from .widgets import (
from .widgets.button import _ButtonBase
class FigureWrapper(param.Parameterized):
    figure = param.Parameter()

    def get_ax(self):
        from matplotlib.backends.backend_agg import FigureCanvas
        from matplotlib.pyplot import Figure
        self.figure = fig = Figure()
        FigureCanvas(fig)
        return fig.subplots()

    def _repr_mimebundle_(self, include=[], exclude=[]):
        return self.layout()._repr_mimebundle_()

    def __panel__(self):
        return self.layout()
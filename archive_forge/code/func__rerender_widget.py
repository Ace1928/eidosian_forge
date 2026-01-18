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
def _rerender_widget(self, p_name):
    watchers = []
    for w in self._internal_callbacks:
        if w.inst is self._widgets[p_name]:
            w.inst.param.unwatch(w)
        else:
            watchers.append(w)
    self._widgets[p_name] = self.widget(p_name)
    self._rerender()
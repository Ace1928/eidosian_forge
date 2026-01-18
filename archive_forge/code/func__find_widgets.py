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
@classmethod
def _find_widgets(cls, op):
    widgets = []
    op_args = list(op['args']) + list(op['kwargs'].values())
    op_args = flatten(op_args)
    for op_arg in op_args:
        if isinstance(op_arg, Widget) and op_arg not in widgets:
            widgets.append(op_arg)
            continue
        if 'ipywidgets' in sys.modules:
            from ipywidgets import Widget as IPyWidget
            if isinstance(op_arg, IPyWidget) and op_arg not in widgets:
                widgets.append(op_arg)
                continue
        if isinstance(op_arg, param.Parameter) and isinstance(op_arg.owner, Widget) and (op_arg.owner not in widgets):
            widgets.append(op_arg.owner)
            continue
        if hasattr(op_arg, '_dinfo'):
            dinfo = op_arg._dinfo
            args = list(dinfo.get('dependencies', []))
            kwargs = dinfo.get('kw', {})
            nested_op = {'args': args, 'kwargs': kwargs}
        elif isinstance(op_arg, slice):
            nested_op = {'args': [op_arg.start, op_arg.stop, op_arg.step], 'kwargs': {}}
        elif isinstance(op_arg, (list, tuple)):
            nested_op = {'args': op_arg, 'kwargs': {}}
        elif isinstance(op_arg, dict):
            nested_op = {'args': (), 'kwargs': op_arg}
        elif isinstance(op_arg, param.rx):
            nested_op = {'args': op_arg._params, 'kwargs': {}}
        else:
            continue
        for widget in cls._find_widgets(nested_op):
            if widget not in widgets:
                widgets.append(widget)
    return widgets
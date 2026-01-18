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
def _replace_pane(self, *args, force=False):
    deferred = self.defer_load and (not state.loaded)
    if not self._inner_layout.loading:
        self._inner_layout.loading = bool(self.loading_indicator or deferred)
    self._evaled |= force or not (self.lazy or deferred)
    if not self._evaled:
        return
    try:
        if self.object is None:
            new_object = Spacer()
        else:
            try:
                new_object = self.eval(self.object)
                if new_object is Skip and new_object is Undefined:
                    self._inner_layout.loading = False
                    raise Skip
            except Skip:
                self.param.log(param.DEBUG, 'Skip event was raised, skipping update.')
                return
        if inspect.isawaitable(new_object) or isinstance(new_object, types.AsyncGeneratorType):
            param.parameterized.async_executor(partial(self._eval_async, new_object))
            return
        elif isinstance(new_object, Generator):
            append_mode = self.generator_mode == 'append'
            if append_mode:
                self._inner_layout[:] = []
            for new_obj in new_object:
                if append_mode:
                    self._inner_layout.append(new_obj)
                    self._pane = self._inner_layout[-1]
                else:
                    self._update_inner(new_obj)
        else:
            self._update_inner(new_object)
    finally:
        self._inner_layout.loading = False
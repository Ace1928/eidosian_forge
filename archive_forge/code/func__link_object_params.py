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
def _link_object_params(self):
    parameterized = get_method_owner(self.object)
    params = parameterized.param.method_dependencies(self.object.__name__)
    deps = params

    def update_pane(*events):
        if any((is_parameterized(event.new) for event in events)):
            new_deps = parameterized.param.method_dependencies(self.object.__name__)
            for p in list(deps):
                if p in new_deps:
                    continue
                watchers = self._internal_callbacks
                for w in list(watchers):
                    if w.inst is p.inst and w.cls is p.cls and (p.name in w.parameter_names):
                        obj = p.cls if p.inst is None else p.inst
                        obj.param.unwatch(w)
                        watchers.remove(w)
                deps.remove(p)
            new_deps = [dep for dep in new_deps if dep not in deps]
            for _, params in full_groupby(new_deps, lambda x: (x.inst or x.cls, x.what)):
                p = params[0]
                pobj = p.cls if p.inst is None else p.inst
                ps = [_p.name for _p in params]
                watcher = pobj.param.watch(update_pane, ps, p.what)
                self._internal_callbacks.append(watcher)
                for p in params:
                    deps.append(p)
        self._replace_pane()
    for _, sub_params in full_groupby(params, lambda x: (x.inst or x.cls, x.what)):
        p = sub_params[0]
        pobj = p.inst or p.cls
        ps = [_p.name for _p in sub_params]
        if isinstance(pobj, Reactive) and self.loading_indicator:
            props = {p: 'loading' for p in ps if p in pobj._linkable_params}
            if props:
                pobj.jslink(self._inner_layout, **props)
        watcher = pobj.param.watch(update_pane, ps, p.what)
        self._internal_callbacks.append(watcher)
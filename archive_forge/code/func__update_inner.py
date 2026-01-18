from __future__ import annotations
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ImportedStyleSheet
from bokeh.models.layouts import (
from .._param import Margin
from ..io.cache import _generate_hash
from ..io.document import create_doc_if_none_exists, unlocked
from ..io.notebook import push
from ..io.state import state
from ..layout.base import (
from ..links import Link
from ..models import ReactiveHTML as _BkReactiveHTML
from ..reactive import Reactive
from ..util import param_reprs, param_watchers
from ..util.checks import is_dataframe, is_series
from ..util.parameters import get_params_to_inherit
from ..viewable import (
def _update_inner(self, new_object: Any) -> None:
    kwargs = dict(self.param.values(), **self._kwargs)
    del kwargs['object']
    new_pane, internal = self._update_from_object(new_object, self._pane, self._internal, **kwargs)
    if new_pane is None:
        return
    self._pane = new_pane
    self._inner_layout[:] = [self._pane]
    self._internal = internal
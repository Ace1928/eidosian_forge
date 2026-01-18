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
@classmethod
def _recursive_update(cls, old: Reactive, new: Reactive):
    """
        Recursively descends through Panel layouts and diffs their
        contents updating only changed parameters ensuring we don't
        have to trigger a full re-render of the entire component.

        Arguments
        ---------
        old: Reactive
          The Reactive component being updated or replaced.
        new: Reactive
          The new Reactive component that the old one is being updated
          or replaced with.
        """
    ignored = ('name',)
    if isinstance(new, ListPanel):
        if len(old) == len(new):
            for i, (sub_old, sub_new) in enumerate(zip(old, new)):
                if type(sub_old) is not type(sub_new):
                    old[i] = new
                    continue
                if isinstance(new, NamedListPanel):
                    old._names[i] = new._names[i]
                cls._recursive_update(sub_old, sub_new)
            ignored += ('objects',)
    pvals = dict(old.param.values())
    new_params = {}
    for p, p_new in new.param.values().items():
        p_old = pvals[p]
        if p in ignored or p_new is p_old:
            continue
        try:
            equal = p_new == p_old
            if is_dataframe(equal) or is_series(equal) or isinstance(equal, np.ndarray):
                equal = equal.all()
            equal = bool(equal)
        except Exception:
            try:
                equal = _generate_hash(p_new) == _generate_hash(p_old)
            except Exception:
                equal = False
        if not equal:
            new_params[p] = p_new
    if isinstance(old, PaneBase):
        changing = any((p in old._rerender_params for p in new_params))
        old._object_changing = changing
        try:
            with param.edit_constant(old):
                old.param.update(**new_params)
        finally:
            old._object_changing = False
    else:
        with param.edit_constant(old):
            old.param.update(**new_params)
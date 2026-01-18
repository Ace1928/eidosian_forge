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
def _update_from_object(cls, object: Any, old_object: Any, was_internal: bool, inplace: bool=False, **kwargs):
    pane_type = cls.get_pane_type(object)
    try:
        links = Link.registry.get(object)
    except TypeError:
        links = []
    custom_watchers = []
    if isinstance(object, Reactive):
        watchers = [w for pwatchers in param_watchers(object).values() for awatchers in pwatchers.values() for w in awatchers]
        custom_watchers = [wfn for wfn in watchers if wfn not in object._internal_callbacks and (not hasattr(wfn.fn, '_watcher_name'))]
    pane, internal = (None, was_internal)
    if type(old_object) is pane_type and (not links and (not custom_watchers) and was_internal or inplace):
        if isinstance(object, Panel) and len(old_object) == len(object):
            for i, (old, new) in enumerate(zip(old_object, object)):
                if type(old) is not type(new):
                    old_object[i] = new
                    continue
                cls._recursive_update(old, new)
        elif isinstance(object, Reactive):
            cls._recursive_update(old_object, object)
        elif old_object.object is not object:
            old_object.object = object
    else:
        pane_params = {k: v for k, v in kwargs.items() if k in pane_type.param}
        pane = panel(object, **pane_params)
        internal = pane is not object
    return (pane, internal)
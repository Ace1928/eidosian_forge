import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _dynamicmap_opts(self, *args, **kwargs):
    from ..util import Dynamic
    clone = kwargs.get('clone', None)
    apply_groups, _, _ = util.deprecated_opts_signature(args, kwargs)
    clone = apply_groups if clone is None else clone
    obj = self._obj if clone else self._obj.clone()
    dmap = Dynamic(obj, operation=lambda obj, **dynkwargs: obj.opts(*args, **kwargs), streams=self._obj.streams, link_inputs=True)
    if not clone:
        with util.disable_constant(self._obj):
            obj.callback = self._obj.callback
            self._obj.callback = dmap.callback
        dmap = self._obj
        dmap.data = dict([(k, v.opts(*args, **kwargs)) for k, v in self._obj.data.items()])
    return dmap
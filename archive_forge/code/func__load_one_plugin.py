import logging
import operator
from . import _cache
from .exception import NoMatches
def _load_one_plugin(self, ep, invoke_on_load, invoke_args, invoke_kwds, verify_requirements):
    if hasattr(ep, 'resolve') and hasattr(ep, 'require'):
        if verify_requirements:
            ep.require()
        plugin = ep.resolve()
    else:
        plugin = ep.load()
    if invoke_on_load:
        obj = plugin(*invoke_args, **invoke_kwds)
    else:
        obj = None
    return Extension(ep.name, ep, plugin, obj)
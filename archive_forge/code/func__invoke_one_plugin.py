import logging
import operator
from . import _cache
from .exception import NoMatches
def _invoke_one_plugin(self, response_callback, func, e, args, kwds):
    try:
        response_callback(func(e, *args, **kwds))
    except Exception as err:
        if self.propagate_map_exceptions:
            raise
        else:
            LOG.error('error calling %r: %s', e.name, err)
            LOG.exception(err)
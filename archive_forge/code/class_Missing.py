import functools
from debugpy.common import json, log, messaging, util
class Missing(object):
    """A dummy component that raises ComponentNotAvailable whenever some
        attribute is accessed on it.
        """
    __getattr__ = __setattr__ = lambda self, *_: report()
    __bool__ = __nonzero__ = lambda self: False
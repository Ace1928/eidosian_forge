import inspect
import logging
from typing import Optional, Union
from ray.util import log_once
from ray.util.annotations import _mark_annotated
def _inner(obj):
    if inspect.isclass(obj):
        obj_init = obj.__init__

        def patched_init(*args, **kwargs):
            if log_once(old or obj.__name__):
                deprecation_warning(old=old or obj.__name__, new=new, help=help, error=error)
            return obj_init(*args, **kwargs)
        obj.__init__ = patched_init
        _mark_annotated(obj)
        return obj

    def _ctor(*args, **kwargs):
        if log_once(old or obj.__name__):
            deprecation_warning(old=old or obj.__name__, new=new, help=help, error=error)
        return obj(*args, **kwargs)
    return _ctor
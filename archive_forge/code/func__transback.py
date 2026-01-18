from .abstract import Thenable
from .promises import promise
def _transback(filter_, callback, args, kwargs, ret):
    try:
        ret = filter_(*args + (ret,), **kwargs)
    except Exception:
        callback.throw()
    else:
        return callback(ret)
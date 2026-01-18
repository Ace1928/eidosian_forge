from functools import wraps
import inspect
import dbus
from . import defer, Deferred, DeferredException
@wraps(func)
def _deferable(*args, **kwargs):

    def on_error(error, deferred):
        if isinstance(error, DeferredException):
            deferred.errback(error)
        else:
            deferred.errback(DeferredException(error))
    try:
        to_defer = kwargs.pop('defer')
    except KeyError:
        stack = inspect.stack()
        try:
            to_defer = stack[2][3] == '_inline_callbacks'
        except IndexError:
            to_defer = False
    if to_defer:
        deferred = Deferred()
        kwargs['reply_handler'] = deferred.callback
        kwargs['error_handler'] = lambda err: on_error(err, deferred)
        func(*args, **kwargs)
        return deferred
    return func(*args, **kwargs)
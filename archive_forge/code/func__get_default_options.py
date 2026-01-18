from sentry_sdk._types import TYPE_CHECKING
def _get_default_options():
    import inspect
    if hasattr(inspect, 'getfullargspec'):
        getargspec = inspect.getfullargspec
    else:
        getargspec = inspect.getargspec
    a = getargspec(ClientConstructor.__init__)
    defaults = a.defaults or ()
    return dict(zip(a.args[-len(defaults):], defaults))
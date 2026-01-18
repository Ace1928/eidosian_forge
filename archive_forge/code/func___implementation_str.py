@staticmethod
def __implementation_str(impl):
    import inspect
    try:
        sig = inspect.signature
        formatsig = str
    except AttributeError:
        sig = inspect.getargspec
        f = inspect.formatargspec
        formatsig = lambda sig: f(*sig)
    try:
        sig = sig(impl)
    except (ValueError, TypeError):
        return repr(impl)
    try:
        name = impl.__qualname__
    except AttributeError:
        name = impl.__name__
    return name + formatsig(sig)
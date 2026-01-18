import functools
import warnings
def _decorate_property(self, prop):
    msg = self.extra

    @property
    @functools.wraps(prop)
    def wrapped(*args, **kwargs):
        warnings.warn(msg, category=FutureWarning)
        return prop.fget(*args, **kwargs)
    return wrapped
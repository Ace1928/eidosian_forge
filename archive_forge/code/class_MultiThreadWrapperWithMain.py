from collections import defaultdict
from threading import get_ident, main_thread
class MultiThreadWrapperWithMain(MultiThreadWrapper):
    """An extension of `MultiThreadWrapper` that exposes the wrapped instance
    corresponding to the [main_thread()](https://docs.python.org/3/library/threading.html#threading.main_thread)
    under the `.main_thread` field.

    This is useful for a falling back to a main instance when needed, but results
    in race conditions if used improperly.
    """

    def __init__(self, base):
        super().__init__(base)

    def __getattr__(self, attr):
        if attr == 'main_thread':
            return self._mtdict[main_thread().ident]
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if attr == 'main_thread':
            raise ValueError('Setting `main_thread` attribute is not allowed')
        else:
            super().__setattr__(attr, value)

    def __dir__(self):
        return super().__dir__() + ['main_thread']
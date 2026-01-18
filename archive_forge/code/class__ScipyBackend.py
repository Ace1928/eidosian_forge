import scipy._lib.uarray as ua
from . import _basic_backend
from . import _realtransforms_backend
from . import _fftlog_backend
class _ScipyBackend:
    """The default backend for fft calculations

    Notes
    -----
    We use the domain ``numpy.scipy`` rather than ``scipy`` because ``uarray``
    treats the domain as a hierarchy. This means the user can install a single
    backend for ``numpy`` and have it implement ``numpy.scipy.fft`` as well.
    """
    __ua_domain__ = 'numpy.scipy.fft'

    @staticmethod
    def __ua_function__(method, args, kwargs):
        fn = getattr(_basic_backend, method.__name__, None)
        if fn is None:
            fn = getattr(_realtransforms_backend, method.__name__, None)
        if fn is None:
            fn = getattr(_fftlog_backend, method.__name__, None)
        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)
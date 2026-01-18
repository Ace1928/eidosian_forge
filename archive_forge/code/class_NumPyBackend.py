import numpy as np
class NumPyBackend:
    """Backend that uses numpy.fft"""
    __ua_domain__ = 'numpy.scipy.fft'

    @staticmethod
    def __ua_function__(method, args, kwargs):
        kwargs.pop('overwrite_x', None)
        fn = getattr(np.fft, method.__name__, None)
        return NotImplemented if fn is None else fn(*args, **kwargs)
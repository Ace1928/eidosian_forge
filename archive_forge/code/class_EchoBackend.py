import numpy as np
class EchoBackend:
    """Backend that just prints the __ua_function__ arguments"""
    __ua_domain__ = 'numpy.scipy.fft'

    @staticmethod
    def __ua_function__(method, args, kwargs):
        print(method, args, kwargs, sep='\n')
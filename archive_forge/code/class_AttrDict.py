import struct
import numpy as np
import tempfile
import zlib
import warnings
class AttrDict(dict):
    """
    A case-insensitive dictionary with access via item, attribute, and call
    notations:

        >>> from scipy.io._idl import AttrDict
        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d.variable
        123
        >>> d('VARIABLE')
        123
        >>> d['missing']
        Traceback (most recent error last):
        ...
        KeyError: 'missing'
        >>> d.missing
        Traceback (most recent error last):
        ...
        AttributeError: 'AttrDict' object has no attribute 'missing'
    """

    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getitem__(self, name):
        return super().__getitem__(name.lower())

    def __setitem__(self, key, value):
        return super().__setitem__(key.lower(), value)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'") from None
    __setattr__ = __setitem__
    __call__ = __getitem__
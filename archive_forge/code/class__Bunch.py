import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
class _Bunch:

    def __init__(self, **kwargs):
        self.__keys = kwargs.keys()
        self.__dict__.update(**kwargs)

    def __repr__(self):
        return '_Bunch({})'.format(', '.join((f'{k}={repr(self.__dict__[k])}' for k in self.__keys)))
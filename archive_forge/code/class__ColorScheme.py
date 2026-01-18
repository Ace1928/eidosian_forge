import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class _ColorScheme(metaclass=abc.ABCMeta):

    @abstractmethod
    def code(self, msg):
        pass

    @abstractmethod
    def errmsg(self, msg):
        pass

    @abstractmethod
    def filename(self, msg):
        pass

    @abstractmethod
    def indicate(self, msg):
        pass

    @abstractmethod
    def highlight(self, msg):
        pass

    @abstractmethod
    def reset(self, msg):
        pass
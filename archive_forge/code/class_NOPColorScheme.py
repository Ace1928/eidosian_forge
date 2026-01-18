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
class NOPColorScheme(_DummyColorScheme):

    def __init__(self, theme=None):
        if theme is not None:
            raise ValueError('specifying a theme has no effect')
        _DummyColorScheme.__init__(self, theme=theme)

    def code(self, msg):
        return msg

    def errmsg(self, msg):
        return msg

    def filename(self, msg):
        return msg

    def indicate(self, msg):
        return msg

    def highlight(self, msg):
        return msg

    def reset(self, msg):
        return msg
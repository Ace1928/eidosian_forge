import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class GeneralTimer(object):

    def __init__(self, fmt, data):
        self.fmt = fmt
        self.data = data

    def report(self):
        _logger.info(self)

    @property
    def obj(self):
        return self.data[-1]

    @property
    def timer(self):
        return self.data[:-1]

    def __str__(self):
        return self.fmt % self.data
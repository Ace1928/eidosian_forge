import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
class CrazyClass(object):

    def __dir__(self):
        return super(CrazyClass, self).__dir__() + ['crazy']

    def __getattr__(self, item):
        if item == 'crazy':
            return lambda x: x
        raise AttributeError(item)
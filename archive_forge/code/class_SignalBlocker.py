import re
import warnings
import weakref
from collections import OrderedDict
from .. import functions as fn
from ..Qt import QtCore
from .ParameterItem import ParameterItem
class SignalBlocker(object):

    def __init__(self, enterFn, exitFn):
        self.enterFn = enterFn
        self.exitFn = exitFn

    def __enter__(self):
        self.enterFn()

    def __exit__(self, exc_type, exc_value, tb):
        self.exitFn()
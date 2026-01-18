import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class NestedInner(object):
    aint = int

    def __init__(self, aint=None):
        self.aint = aint
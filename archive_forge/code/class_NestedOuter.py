import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class NestedOuter(object):
    inner = NestedInner
    inner_array = wsme.types.wsattr([NestedInner])
    inner_dict = {wsme.types.text: NestedInner}

    def __init__(self):
        self.inner = NestedInner(0)
import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class NamedAttrsObject(object):

    def __init__(self, v1=Unset, v2=Unset):
        self.attr_1 = v1
        self.attr_2 = v2
    attr_1 = wsme.types.wsattr(int, name='attr.1')
    attr_2 = wsme.types.wsattr(int, name='attr.2')
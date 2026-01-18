import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class FakeCounter(object):

    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 4:
            self.n += 1
            return self.n
        else:
            raise StopIteration
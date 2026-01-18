import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class Raiser(object):

    def __get__(self, obj, type=None):
        if obj is None:
            raise AttributeError('Can only be accessed via an instance')
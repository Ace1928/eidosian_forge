import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
class SubClass(Mock):

    def _get(self):
        return 3

    def _set(self, value):
        raise NameError('strange error')
    some_attribute = property(_get, _set)
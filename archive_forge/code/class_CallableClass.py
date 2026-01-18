import functools
from oslotest import base as test_base
from oslo_utils import reflection
class CallableClass(object):

    def __call__(self, i, j):
        pass
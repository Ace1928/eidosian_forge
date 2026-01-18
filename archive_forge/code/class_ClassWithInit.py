import functools
from oslotest import base as test_base
from oslo_utils import reflection
class ClassWithInit(object):

    def __init__(self, k, lll):
        pass
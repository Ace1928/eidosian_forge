from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
class ThirdParty(object):

    def __init__(self, foo=b'bar'):
        self.foo = foo

    def __eq__(self, other):
        return isinstance(other, ThirdParty) and self.foo == other.foo
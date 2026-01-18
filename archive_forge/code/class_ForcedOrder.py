import re
import unittest
from wsme import exc
from wsme import types
class ForcedOrder(object):
    _wsme_attr_order = ('a2', 'a1', 'a3')
    a1 = int
    a2 = int
    a3 = int
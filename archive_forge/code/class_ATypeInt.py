import re
import unittest
from wsme import exc
from wsme import types
class ATypeInt(object):
    attr = types.IntegerType(minimum=1, maximum=5)
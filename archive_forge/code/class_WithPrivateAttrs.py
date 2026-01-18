import re
import unittest
from wsme import exc
from wsme import types
class WithPrivateAttrs(object):
    _private = 12
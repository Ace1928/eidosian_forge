import re
import unittest
from wsme import exc
from wsme import types
class ABCDType(object):
    a_list = types.wsattr([int], name='a.list')
    astr = str
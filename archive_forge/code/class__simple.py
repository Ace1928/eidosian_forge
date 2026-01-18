import inspect
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.core.base.disable_methods import disable_methods
from pyomo.common.modeling import NOTSET
class _simple(object):

    def __init__(self, name):
        self.name = name
        self._d = 'd'
        self._e = 'e'

    def construct(self, data=None):
        return 'construct'

    def a(self):
        return 'a'

    def b(self):
        return 'b'

    def c(self):
        return 'c'

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value='d'):
        self._d = value

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value='e'):
        self._e = value

    def f(self, arg1, arg2=1, arg3=NOTSET, arg4=local_instance):
        return 'f:%s,%s,%s,%s' % (arg1, arg2, arg3, arg4)

    @property
    def g(self):
        return 'g'

    @property
    def h(self):
        return 'h'
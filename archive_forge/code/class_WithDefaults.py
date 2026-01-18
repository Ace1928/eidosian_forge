from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
class WithDefaults(object):

    @decorators.SetParseFns(float)
    def example1(self, arg1=10):
        return (arg1, type(arg1))

    @decorators.SetParseFns(arg1=float)
    def example2(self, arg1=10):
        return (arg1, type(arg1))
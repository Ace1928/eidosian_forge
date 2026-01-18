import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class NumericLabeler(object):

    def __init__(self, prefix, start=0):
        self.id = start
        self.prefix = prefix

    def __call__(self, obj=None):
        self.id += 1
        return self.prefix + str(self.id)

    @deprecated("The 'remove_obj' method is no longer necessary now that 'getname' does not support the use of a name buffer", version='6.4.1')
    def remove_obj(self, obj):
        pass
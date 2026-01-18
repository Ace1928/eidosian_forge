import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
class AlphaNumericTextLabeler(object):

    def __call__(self, obj):
        return alphanum_label_from_name(obj.getname(True))
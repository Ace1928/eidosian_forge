import logging
from pyomo.common import deprecated
from pyomo.core.base import Transformation, TransformationFactory, Var, Suffix, Reals
@TransformationFactory.register('core.fix_discrete', doc='[DEPRECATED] Fix all integer variables to their current values')
@deprecated('core.fix_discrete is deprecated.  Use core.fix_integer_vars', version='5.7')
class FixDiscreteVars(FixIntegerVars):

    def __init__(self, **kwds):
        super(FixDiscreteVars, self).__init__(**kwds)
from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _convert_M_to_tuple(M, constraint, disjunct=None):
    if not isinstance(M, (tuple, list)):
        if M is None:
            M = (None, None)
        else:
            try:
                M = (-M, M)
            except:
                logger.error('Error converting scalar M-value %s to (-M,M).  Is %s not a numeric type?' % (M, type(M)))
                raise
    if len(M) != 2:
        constraint_name = constraint.name
        if disjunct is not None:
            constraint_name += ' relative to Disjunct %s' % disjunct.name
        raise GDP_Error('Big-M %s for constraint %s is not of length two. Expected either a single value or tuple or list of length two specifying M values for the lower and upper sides of the constraint respectively.' % (str(M), constraint.name))
    return M
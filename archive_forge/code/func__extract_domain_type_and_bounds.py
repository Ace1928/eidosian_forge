from pyomo.common.modeling import NoArgumentGiven
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericValue, is_numeric_data, value
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.set_types import RealSet, IntegerSet
def _extract_domain_type_and_bounds(domain_type, domain, lb, ub):
    if domain is not None:
        if domain_type is not None:
            raise ValueError("At most one of the 'domain' and 'domain_type' keywords can be changed from their default value when initializing a variable.")
        domain_lb, domain_ub, domain_step = domain.get_interval()
        if domain_step == 0:
            domain_type = RealSet
        elif domain_step == 1:
            domain_type = IntegerSet
        if domain_lb is not None:
            if lb is not None:
                raise ValueError("The 'lb' keyword can not be used to initialize a variable when the domain lower bound is finite.")
            lb = domain_lb
        if domain_ub is not None:
            if ub is not None:
                raise ValueError("The 'ub' keyword can not be used to initialize a variable when the domain upper bound is finite.")
            ub = domain_ub
    elif domain_type is None:
        domain_type = RealSet
    if domain_type not in IVariable._valid_domain_types:
        raise ValueError("Domain type '%s' is not valid. Must be one of: %s" % (domain_type, IVariable._valid_domain_types))
    return (domain_type, lb, ub)
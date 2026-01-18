from pyomo.common.deprecation import RenamedClass
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.indexed_component import rule_wrapper
from pyomo.core.base.expression import (
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
def _trap_rule(rule, m, *a):
    ds = sorted(m.find_component(wrt.local_name))
    return sum((0.5 * (ds[i + 1] - ds[i]) * (rule(m, *a[0:loc] + (ds[i + 1],) + a[loc:]) + rule(m, *a[0:loc] + (ds[i],) + a[loc:])) for i in range(len(ds) - 1)))
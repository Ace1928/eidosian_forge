import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def _find_deps_any_in_init(self, unresolved_value):
    deps = function.dependencies(unresolved_value)
    if any((res.action == res.INIT for res in deps)):
        return True
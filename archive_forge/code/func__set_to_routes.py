from operator import itemgetter
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
def _set_to_routes(route_set):
    """The reverse of _routes_to_set.

    _set_to_routes(_routes_to_set(routes)) == routes
    """
    return [dict(r) for r in route_set]
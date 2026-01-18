from operator import itemgetter
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
def _routes_to_set(route_list):
    """Convert routes to a set that can be diffed.

    Convert the in-API/in-template routes format to another data type that
    has the same information content but that is hashable, so we can put
    routes in a set and perform set operations on them.
    """
    return set((frozenset(r.items()) for r in route_list))
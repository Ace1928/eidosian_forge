from operator import itemgetter
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
def _raise_if_duplicate(router_existing, routes_to_add):
    """Detect trying to add duplicate routes in create/update

    Take the response of show_router() for an existing router and a list of
    routes to add and raise PhysicalResourceExists if we try to add a route
    already existing on the router. Otherwise do not raise and return None.

    You cannot use this to detect duplicate routes atomically while adding
    a route so when you use this you'll inevitably create race conditions.
    """
    routes_existing = _routes_to_set(router_existing['router']['routes'])
    for route in _routes_to_set(routes_to_add):
        if route in routes_existing:
            original = _set_to_routes(set([route]))
            name = _generate_name(router, original)
            raise exception.PhysicalResourceExists(name=name)
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import port
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
from heat.engine import translation
A resource for creating port forwarding for floating IPs.

    This resource creates port forwarding for floating IPs.
    These are sub-resource of exsisting Floating ips, which requires the
    service_plugin and extension port_forwarding enabled and that the floating
    ip is not associated with a neutron port.
    
import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class RemoveInterfaceRouter(RouterInterfaceCommand):
    """Remove an internal network interface from a router."""

    def call_api(self, neutron_client, router_id, body):
        return neutron_client.remove_interface_router(router_id, body)

    def success_message(self, router_id, portinfo):
        return _('Removed interface from router %s.') % router_id
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class ShowEndpointGroup(neutronv20.ShowCommand):
    """Show a specific VPN endpoint group."""
    resource = 'endpoint_group'
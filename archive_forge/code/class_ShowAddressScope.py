from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowAddressScope(neutronV20.ShowCommand):
    """Show information about an address scope."""
    resource = 'address_scope'
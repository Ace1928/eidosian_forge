from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class DeleteVip(neutronV20.DeleteCommand):
    """Delete a given vip."""
    resource = 'vip'
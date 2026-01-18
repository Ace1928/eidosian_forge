from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class UpdateVip(neutronV20.UpdateCommand):
    """Update a given vip."""
    resource = 'vip'
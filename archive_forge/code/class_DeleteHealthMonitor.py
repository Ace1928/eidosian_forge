from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteHealthMonitor(neutronV20.DeleteCommand):
    """LBaaS v2 Delete a given healthmonitor."""
    resource = 'healthmonitor'
    shadow_resource = 'lbaas_healthmonitor'
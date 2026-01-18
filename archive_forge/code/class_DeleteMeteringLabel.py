from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class DeleteMeteringLabel(neutronv20.DeleteCommand):
    """Delete a given metering label."""
    resource = 'metering_label'
    allow_names = True
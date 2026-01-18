from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ShowMeteringLabel(neutronv20.ShowCommand):
    """Show information of a given metering label."""
    resource = 'metering_label'
    allow_names = True
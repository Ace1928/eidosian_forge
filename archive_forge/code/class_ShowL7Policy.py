from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowL7Policy(neutronV20.ShowCommand):
    """LBaaS v2 Show information of a given L7 policy."""
    resource = 'l7policy'
    shadow_resource = 'lbaas_l7policy'
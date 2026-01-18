from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteL7Rule(LbaasL7RuleMixin, neutronV20.DeleteCommand):
    """LBaaS v2 Delete a given L7 rule."""
    resource = 'rule'
    shadow_resource = 'lbaas_l7rule'
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_policy_id(client, policy_id_or_name):
    return neutronV20.find_resourceid_by_name_or_id(client, 'l7policy', policy_id_or_name, cmd_resource='lbaas_l7policy')
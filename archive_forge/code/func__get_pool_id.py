from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _get_pool_id(client, pool_id_or_name):
    return neutronV20.find_resourceid_by_name_or_id(client, 'pool', pool_id_or_name, cmd_resource='lbaas_pool')
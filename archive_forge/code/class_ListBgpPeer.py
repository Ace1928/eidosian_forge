from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class ListBgpPeer(command.Lister):
    _description = _('List BGP peers')

    def take_action(self, parsed_args):
        data = self.app.client_manager.network.bgp_peers(retrieve_all=True)
        headers = ('ID', 'Name', 'Peer IP', 'Remote AS')
        columns = ('id', 'name', 'peer_ip', 'remote_as')
        return (headers, (utils.get_dict_properties(s, columns) for s in data))
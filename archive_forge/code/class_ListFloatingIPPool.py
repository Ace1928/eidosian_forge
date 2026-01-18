from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ListFloatingIPPool(common.NetworkAndComputeLister):
    _description = _('List pools of floating IP addresses')

    def take_action_network(self, client, parsed_args):
        msg = _('Floating ip pool operations are only available for Compute v2 network.')
        raise exceptions.CommandError(msg)

    def take_action_compute(self, client, parsed_args):
        columns = ('Name',)
        data = client.api.floating_ip_pool_list()
        return (columns, (utils.get_dict_properties(s, columns) for s in data))
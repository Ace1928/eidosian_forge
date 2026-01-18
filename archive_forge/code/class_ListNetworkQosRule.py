import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ListNetworkQosRule(command.Lister):
    _description = _('List Network QoS rules')

    def get_parser(self, prog_name):
        parser = super(ListNetworkQosRule, self).get_parser(prog_name)
        parser.add_argument('qos_policy', metavar='<qos-policy>', help=_('QoS policy that contains the rule (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'qos_policy_id', 'type', 'max_kbps', 'max_burst_kbps', 'min_kbps', 'min_kpps', 'dscp_mark', 'direction')
        column_headers = ('ID', 'QoS Policy ID', 'Type', 'Max Kbps', 'Max Burst Kbits', 'Min Kbps', 'Min Kpps', 'DSCP mark', 'Direction')
        qos = client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False)
        data = qos.rules
        return (column_headers, (_get_item_properties(s, columns) for s in data))
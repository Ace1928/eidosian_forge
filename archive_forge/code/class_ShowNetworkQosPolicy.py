import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ShowNetworkQosPolicy(command.ShowOne):
    _description = _('Display QoS policy details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkQosPolicy, self).get_parser(prog_name)
        parser.add_argument('policy', metavar='<qos-policy>', help=_('QoS policy to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_qos_policy(parsed_args.policy, ignore_missing=False)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)
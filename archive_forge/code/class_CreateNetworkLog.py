import copy
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as fwaas_const
class CreateNetworkLog(command.ShowOne):
    _description = _('Create a new network log')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkLog, self).get_parser(prog_name)
        _get_common_parser(parser)
        osc_utils.add_project_owner_option_to_parser(parser)
        parser.add_argument('name', metavar='<name>', help=_('Name for the network log'))
        parser.add_argument('--event', metavar='{ALL,ACCEPT,DROP}', choices=['ALL', 'ACCEPT', 'DROP'], type=nc_utils.convert_to_uppercase, help=_('An event to store with log'))
        parser.add_argument('--resource-type', metavar='<resource-type>', required=True, type=nc_utils.convert_to_lowercase, help=_('Network log type(s). You can see supported type(s) with following command:\n$ openstack network loggable resources list'))
        parser.add_argument('--resource', metavar='<resource>', help=_('Name or ID of resource (security group or firewall group) that used for logging. You can control for logging target combination with --target option.'))
        parser.add_argument('--target', metavar='<target>', help=_('Port (name or ID) for logging. You can control for logging target combination with --resource option.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.neutronclient
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        obj = client.create_network_log({'log': attrs})['log']
        columns, display_columns = column_util.get_columns(obj, _attr_map)
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)
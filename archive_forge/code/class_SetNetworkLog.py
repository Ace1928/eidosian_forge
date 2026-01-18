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
class SetNetworkLog(command.Command):
    _description = _('Set network log properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkLog, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument('network_log', metavar='<network-log>', help=_('Network log to set (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of the network log'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.neutronclient
        log_id = client.find_resource('log', parsed_args.network_log, cmd_resource=NET_LOG)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_network_log(log_id, {'log': attrs})
        except Exception as e:
            msg = _("Failed to set network log '%(logging)s': %(e)s") % {'logging': parsed_args.network_log, 'e': e}
            raise exceptions.CommandError(msg)
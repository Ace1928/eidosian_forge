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
class DeleteNetworkLog(command.Command):
    _description = _('Delete network log(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkLog, self).get_parser(prog_name)
        parser.add_argument('network_log', metavar='<network-log>', nargs='+', help=_('Network log(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.neutronclient
        result = 0
        for log_res in parsed_args.network_log:
            try:
                log_id = client.find_resource('log', log_res, cmd_resource=NET_LOG)['id']
                client.delete_network_log(log_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete network log with name or ID '%(network_log)s': %(e)s"), {'network_log': log_res, 'e': e})
        if result > 0:
            total = len(parsed_args.network_log)
            msg = _('%(result)s of %(total)s network log(s) failed to delete') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
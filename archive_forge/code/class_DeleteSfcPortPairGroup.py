import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class DeleteSfcPortPairGroup(command.Command):
    _description = _('Delete a given port pair group')

    def get_parser(self, prog_name):
        parser = super(DeleteSfcPortPairGroup, self).get_parser(prog_name)
        parser.add_argument('port_pair_group', metavar='<port-pair-group>', nargs='+', help=_('Port pair group(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for ppg in parsed_args.port_pair_group:
            try:
                ppg_id = client.find_sfc_port_pair_group(ppg, ignore_missing=False)['id']
                client.delete_sfc_port_pair_group(ppg_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete port pair group with name or ID '%(ppg)s': %(e)s"), {'ppg': ppg, 'e': e})
        if result > 0:
            total = len(parsed_args.port_pair_group)
            msg = _('%(result)s of %(total)s port pair group(s) failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class DeleteSfcPortChain(command.Command):
    _description = _('Delete a given port chain')

    def get_parser(self, prog_name):
        parser = super(DeleteSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('port_chain', metavar='<port-chain>', nargs='+', help=_('Port chain(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for pc in parsed_args.port_chain:
            try:
                pc_id = client.find_sfc_port_chain(pc, ignore_missing=False)['id']
                client.delete_sfc_port_chain(pc_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete port chain with name or ID '%(pc)s': %(e)s"), {'pc': pc, 'e': e})
        if result > 0:
            total = len(parsed_args.port_chain)
            msg = _('%(result)s of %(total)s port chain(s) failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
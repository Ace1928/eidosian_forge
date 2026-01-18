import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class DeleteSfcPortPair(command.Command):
    _description = _('Delete a given port pair')

    def get_parser(self, prog_name):
        parser = super(DeleteSfcPortPair, self).get_parser(prog_name)
        parser.add_argument('port_pair', metavar='<port-pair>', nargs='+', help=_('Port pair(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for pp in parsed_args.port_pair:
            try:
                port_pair_id = client.find_sfc_port_pair(pp, ignore_missing=False)['id']
                client.delete_sfc_port_pair(port_pair_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete port pair with name or ID '%(port_pair)s': %(e)s"), {'port_pair': pp, 'e': e})
        if result > 0:
            total = len(parsed_args.port_pair)
            msg = _('%(result)s of %(total)s port pair(s) failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
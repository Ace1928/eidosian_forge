import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import identity as identity_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from openstackclient.i18n import _
class DeleteNetworkTrunk(command.Command):
    """Delete a given network trunk"""

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkTrunk, self).get_parser(prog_name)
        parser.add_argument('trunk', metavar='<trunk>', nargs='+', help=_('Trunk(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for trunk in parsed_args.trunk:
            try:
                trunk_id = client.find_trunk(trunk).id
                client.delete_trunk(trunk_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete trunk with name or ID '%(trunk)s': %(e)s"), {'trunk': trunk, 'e': e})
        if result > 0:
            total = len(parsed_args.trunk)
            msg = _('%(result)s of %(total)s trunks failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
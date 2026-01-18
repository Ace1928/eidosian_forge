import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteCredential(command.Command):
    _description = _('Delete credential(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteCredential, self).get_parser(prog_name)
        parser.add_argument('credential', metavar='<credential-id>', nargs='+', help=_('ID of credential(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.credential:
            try:
                identity_client.credentials.delete(i)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete credentials with ID '%(credential)s': %(e)s"), {'credential': i, 'e': e})
        if result > 0:
            total = len(parsed_args.credential)
            msg = _('%(result)s of %(total)s credential failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
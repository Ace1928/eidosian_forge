import datetime
import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteApplicationCredential(command.Command):
    _description = _('Delete application credentials(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteApplicationCredential, self).get_parser(prog_name)
        parser.add_argument('application_credential', metavar='<application-credential>', nargs='+', help=_('Application credentials(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for ac in parsed_args.application_credential:
            try:
                app_cred = utils.find_resource(identity_client.application_credentials, ac)
                identity_client.application_credentials.delete(app_cred.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete application credential with name or ID '%(ac)s': %(e)s"), {'ac': ac, 'e': e})
        if errors > 0:
            total = len(parsed_args.application_credential)
            msg = _('%(errors)s of %(total)s application credentials failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)
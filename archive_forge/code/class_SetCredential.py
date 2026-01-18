import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class SetCredential(command.Command):
    _description = _('Set credential properties')

    def get_parser(self, prog_name):
        parser = super(SetCredential, self).get_parser(prog_name)
        parser.add_argument('credential', metavar='<credential-id>', help=_('ID of credential to change'))
        parser.add_argument('--user', metavar='<user>', required=True, help=_('User that owns the credential (name or ID)'))
        parser.add_argument('--type', metavar='<type>', required=True, help=_('New credential type: cert, ec2, totp and so on'))
        parser.add_argument('--data', metavar='<data>', required=True, help=_('New credential data'))
        parser.add_argument('--project', metavar='<project>', help=_('Project which limits the scope of the credential (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        user_id = utils.find_resource(identity_client.users, parsed_args.user).id
        if parsed_args.project:
            project = utils.find_resource(identity_client.projects, parsed_args.project).id
        else:
            project = None
        identity_client.credentials.update(parsed_args.credential, user=user_id, type=parsed_args.type, blob=parsed_args.data, project=project)
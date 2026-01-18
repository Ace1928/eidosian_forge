from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class CreateRequestToken(command.ShowOne):
    _description = _('Create a request token')

    def get_parser(self, prog_name):
        parser = super(CreateRequestToken, self).get_parser(prog_name)
        parser.add_argument('--consumer-key', metavar='<consumer-key>', help=_('Consumer key (required)'), required=True)
        parser.add_argument('--consumer-secret', metavar='<consumer-secret>', help=_('Consumer secret (required)'), required=True)
        parser.add_argument('--project', metavar='<project>', help=_('Project that consumer wants to access (name or ID) (required)'), required=True)
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain owning <project> (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        if parsed_args.domain:
            domain = common.find_domain(identity_client, parsed_args.domain)
            project = utils.find_resource(identity_client.projects, parsed_args.project, domain_id=domain.id)
        else:
            project = utils.find_resource(identity_client.projects, parsed_args.project)
        token_client = identity_client.oauth1.request_tokens
        request_token = token_client.create(parsed_args.consumer_key, parsed_args.consumer_secret, project.id)
        return zip(*sorted(request_token._info.items()))
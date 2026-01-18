import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class ShowRegisteredLimit(command.ShowOne):
    _description = _('Display registered limit details')

    def get_parser(self, prog_name):
        parser = super(ShowRegisteredLimit, self).get_parser(prog_name)
        parser.add_argument('registered_limit_id', metavar='<registered-limit-id>', help=_('Registered limit to display (ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        registered_limit = identity_client.registered_limits.get(parsed_args.registered_limit_id)
        registered_limit._info.pop('links', None)
        return zip(*sorted(registered_limit._info.items()))
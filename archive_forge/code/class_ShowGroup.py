import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowGroup(command.ShowOne):
    _description = _('Display group details')

    def get_parser(self, prog_name):
        parser = super(ShowGroup, self).get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Group to display (name or ID)'))
        parser.add_argument('--domain', metavar='<domain>', help=_('Domain containing <group> (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        group = common.find_group(identity_client, parsed_args.group, domain_name_or_id=parsed_args.domain)
        group._info.pop('links')
        return zip(*sorted(group._info.items()))
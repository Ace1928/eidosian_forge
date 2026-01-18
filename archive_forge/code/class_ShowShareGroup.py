import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ShowShareGroup(command.ShowOne):
    """Show share group."""
    _description = _('Show details about a share group')

    def get_parser(self, prog_name):
        parser = super(ShowShareGroup, self).get_parser(prog_name)
        parser.add_argument('share_group', metavar='<share-group>', help=_('Name or ID of the share group.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group = osc_utils.find_resource(share_client.share_groups, parsed_args.share_group)
        printable_share_group = share_group._info
        printable_share_group.pop('links', None)
        if printable_share_group.get('share_types'):
            if parsed_args.formatter == 'table':
                printable_share_group['share_types'] = '\n'.join(printable_share_group['share_types'])
        return self.dict2columns(printable_share_group)
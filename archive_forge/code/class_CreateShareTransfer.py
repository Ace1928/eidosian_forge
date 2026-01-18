import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class CreateShareTransfer(command.ShowOne):
    """Create a new share transfer."""
    _description = _('Create a new share transfer')

    def get_parser(self, prog_name):
        parser = super(CreateShareTransfer, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help='Name or ID of share to transfer.')
        parser.add_argument('--name', metavar='<name>', default=None, help='Transfer name. Default=None.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = osc_utils.find_resource(share_client.shares, parsed_args.share)
        transfer = share_client.transfers.create(share.id, name=parsed_args.name)
        transfer._info.pop('links', None)
        return self.dict2columns(transfer._info)
import logging
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowTransferRequest(command.ShowOne):
    _description = _('Show volume transfer request details.')

    def get_parser(self, prog_name):
        parser = super(ShowTransferRequest, self).get_parser(prog_name)
        parser.add_argument('transfer_request', metavar='<transfer-request>', help=_('Volume transfer request to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume_transfer_request = utils.find_resource(volume_client.transfers, parsed_args.transfer_request)
        volume_transfer_request._info.pop('links', None)
        return zip(*sorted(volume_transfer_request._info.items()))
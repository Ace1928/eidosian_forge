import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShowMessage(command.ShowOne):
    """Show details about a message."""
    _description = _('Show details about a message')

    def get_parser(self, prog_name):
        parser = super(ShowMessage, self).get_parser(prog_name)
        parser.add_argument('message', metavar='<message>', help=_('ID of the message.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        message = apiutils.find_resource(share_client.messages, parsed_args.message)
        return (MESSAGE_ATTRIBUTES, oscutils.get_dict_properties(message._info, MESSAGE_ATTRIBUTES))
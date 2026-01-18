import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class DeleteMessage(command.Command):
    """Remove one or more messages."""
    _description = _('Remove one or more messages')

    def get_parser(self, prog_name):
        parser = super(DeleteMessage, self).get_parser(prog_name)
        parser.add_argument('message', metavar='<message>', nargs='+', help=_('ID of the message(s).'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        failure_count = 0
        for message in parsed_args.message:
            try:
                message_ref = apiutils.find_resource(share_client.messages, message)
                share_client.messages.delete(message_ref)
            except Exception as e:
                failure_count += 1
                LOG.error(_('Delete for message %(message)s failed: %(e)s'), {'message': message, 'e': e})
        if failure_count > 0:
            raise exceptions.CommandError(_('Unable to delete some or all of the specified messages.'))
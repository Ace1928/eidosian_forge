import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CompleteVolumeAttachment(command.Command):
    """Complete an attachment for a volume."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('attachment', metavar='<attachment>', help=_('ID of volume attachment to mark as completed'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.44'):
            msg = _("--os-volume-api-version 3.44 or greater is required to support the 'volume attachment complete' command")
            raise exceptions.CommandError(msg)
        volume_client.attachments.complete(parsed_args.attachment)
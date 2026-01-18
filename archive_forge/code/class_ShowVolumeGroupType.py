import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowVolumeGroupType(command.ShowOne):
    """Show detailed information for a volume group type.

    This command requires ``--os-volume-api-version`` 3.11 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group_type', metavar='<group_type>', help=_('Name or ID of volume group type.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.11'):
            msg = _("--os-volume-api-version 3.11 or greater is required to support the 'volume group type show' command")
            raise exceptions.CommandError(msg)
        group_type = utils.find_resource(volume_client.group_types, parsed_args.group)
        return _format_group_type(group_type)
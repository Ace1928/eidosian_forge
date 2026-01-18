import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateVolumeGroupType(command.ShowOne):
    """Create a volume group type.

    This command requires ``--os-volume-api-version`` 3.11 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of new volume group type.'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of the volume group type.'))
        type_group = parser.add_mutually_exclusive_group()
        type_group.add_argument('--public', dest='is_public', action='store_true', default=True, help=_('Volume group type is available to other projects (default)'))
        type_group.add_argument('--private', dest='is_public', action='store_false', help=_('Volume group type is not available to other projects'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.11'):
            msg = _("--os-volume-api-version 3.11 or greater is required to support the 'volume group type create' command")
            raise exceptions.CommandError(msg)
        group_type = volume_client.group_types.create(parsed_args.name, parsed_args.description, parsed_args.is_public)
        return _format_group_type(group_type)
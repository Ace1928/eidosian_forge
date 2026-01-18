import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class SetVolumeAttachment(command.ShowOne):
    """Update an attachment for a volume.

    This call is designed to be more of an volume attachment completion than
    anything else. It expects the value of a connector object to notify the
    driver that the volume is going to be connected and where it's being
    connected to.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('attachment', metavar='<attachment>', help=_('ID of volume attachment.'))
        parser.add_argument('--initiator', metavar='<initiator>', help=_('IQN of the initiator attaching to'))
        parser.add_argument('--ip', metavar='<ip>', help=_('IP of the system attaching to'))
        parser.add_argument('--host', metavar='<host>', help=_('Name of the host attaching to'))
        parser.add_argument('--platform', metavar='<platform>', help=_('Platform type'))
        parser.add_argument('--os-type', metavar='<ostype>', help=_('OS type'))
        parser.add_argument('--multipath', action='store_true', dest='multipath', default=False, help=_('Use multipath'))
        parser.add_argument('--no-multipath', action='store_false', dest='multipath', help=_('Use multipath'))
        parser.add_argument('--mountpoint', metavar='<mountpoint>', help=_('Mountpoint volume will be attached at'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.27'):
            msg = _("--os-volume-api-version 3.27 or greater is required to support the 'volume attachment set' command")
            raise exceptions.CommandError(msg)
        connector = {'initiator': parsed_args.initiator, 'ip': parsed_args.ip, 'platform': parsed_args.platform, 'host': parsed_args.host, 'os_type': parsed_args.os_type, 'multipath': parsed_args.multipath, 'mountpoint': parsed_args.mountpoint}
        attachment = volume_client.attachments.update(parsed_args.attachment, connector)
        return _format_attachment(attachment)
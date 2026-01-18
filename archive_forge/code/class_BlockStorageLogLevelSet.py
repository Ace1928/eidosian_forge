from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class BlockStorageLogLevelSet(command.Command):
    """Set log level of block storage service

    Supported by --os-volume-api-version 3.32 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('level', metavar='<log-level>', choices=('INFO', 'WARNING', 'ERROR', 'DEBUG'), type=str.upper, help=_('Desired log level.'))
        parser.add_argument('--host', metavar='<host>', default='', help=_('Set block storage service log level of specified host (name only)'))
        parser.add_argument('--service', metavar='<service>', default='', choices=('', '*', 'cinder-api', 'cinder-volume', 'cinder-scheduler', 'cinder-backup'), help=_('Set block storage service log level of specified service (name only)'))
        parser.add_argument('--log-prefix', metavar='<log-prefix>', default='', help="Prefix for the log, e.g. 'sqlalchemy'")
        return parser

    def take_action(self, parsed_args):
        service_client = self.app.client_manager.volume
        if service_client.api_version < api_versions.APIVersion('3.32'):
            msg = _("--os-volume-api-version 3.32 or greater is required to support the 'block storage log level set' command")
            raise exceptions.CommandError(msg)
        service_client.services.set_log_levels(level=parsed_args.level, binary=parsed_args.service, server=parsed_args.host, prefix=parsed_args.log_prefix)
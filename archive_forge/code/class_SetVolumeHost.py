from osc_lib.command import command
from openstackclient.i18n import _
class SetVolumeHost(command.Command):
    _description = _('Set volume host properties')

    def get_parser(self, prog_name):
        parser = super(SetVolumeHost, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host-name>', help=_('Name of volume host'))
        enabled_group = parser.add_mutually_exclusive_group()
        enabled_group.add_argument('--disable', action='store_true', help=_('Freeze and disable the specified volume host'))
        enabled_group.add_argument('--enable', action='store_true', help=_('Thaw and enable the specified volume host'))
        return parser

    def take_action(self, parsed_args):
        service_client = self.app.client_manager.volume
        if parsed_args.enable:
            service_client.services.thaw_host(parsed_args.host)
        if parsed_args.disable:
            service_client.services.freeze_host(parsed_args.host)
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class SetHost(command.Command):
    _description = _('Set host properties')

    def get_parser(self, prog_name):
        parser = super(SetHost, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help=_('Host to modify (name only)'))
        status = parser.add_mutually_exclusive_group()
        status.add_argument('--enable', action='store_true', help=_('Enable the host'))
        status.add_argument('--disable', action='store_true', help=_('Disable the host'))
        maintenance = parser.add_mutually_exclusive_group()
        maintenance.add_argument('--enable-maintenance', action='store_true', help=_('Enable maintenance mode for the host'))
        maintenance.add_argument('--disable-maintenance', action='store_true', help=_('Disable maintenance mode for the host'))
        return parser

    def take_action(self, parsed_args):
        kwargs = {}
        if parsed_args.enable:
            kwargs['status'] = 'enable'
        if parsed_args.disable:
            kwargs['status'] = 'disable'
        if parsed_args.enable_maintenance:
            kwargs['maintenance_mode'] = 'enable'
        if parsed_args.disable_maintenance:
            kwargs['maintenance_mode'] = 'disable'
        compute_client = self.app.client_manager.compute
        compute_client.api.host_set(parsed_args.host, **kwargs)
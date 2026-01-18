from keystoneauth1.loading import base
from osc_lib.command import command
from openstackclient.i18n import _
class ShowConfiguration(command.ShowOne):
    _description = _('Display configuration details')
    auth_required = False

    def get_parser(self, prog_name):
        parser = super(ShowConfiguration, self).get_parser(prog_name)
        mask_group = parser.add_mutually_exclusive_group()
        mask_group.add_argument('--mask', dest='mask', action='store_true', default=True, help=_('Attempt to mask passwords (default)'))
        mask_group.add_argument('--unmask', dest='mask', action='store_false', help=_('Show password in clear text'))
        return parser

    def take_action(self, parsed_args):
        info = self.app.client_manager.get_configuration()
        secret_opts = ['password', 'token']
        if getattr(self.app.client_manager, 'auth_plugin_name', None):
            auth_plg_name = self.app.client_manager.auth_plugin_name
            secret_opts = [o.dest for o in base.get_plugin_options(auth_plg_name) if o.secret]
        for key, value in info.pop('auth', {}).items():
            if parsed_args.mask and key.lower() in secret_opts:
                value = REDACTED
            info['auth.' + key] = value
        if parsed_args.mask:
            for secret_opt in secret_opts:
                if secret_opt in info:
                    info[secret_opt] = REDACTED
        return zip(*sorted(info.items()))
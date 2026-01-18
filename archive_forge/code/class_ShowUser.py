import functools
import logging
from cliff import columns as cliff_columns
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowUser(command.ShowOne):
    _description = _('Display user details')

    def get_parser(self, prog_name):
        parser = super(ShowUser, self).get_parser(prog_name)
        parser.add_argument('user', metavar='<user>', help=_('User to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        info = {}
        try:
            user = utils.find_resource(identity_client.users, parsed_args.user)
            info.update(user._info)
        except ks_exc.Forbidden:
            auth_ref = self.app.client_manager.auth_ref
            if parsed_args.user == auth_ref.user_id or parsed_args.user == auth_ref.username:
                info = {'id': auth_ref.user_id, 'name': auth_ref.username, 'project_id': auth_ref.project_id, 'enabled': True}
            else:
                raise
        if 'tenantId' in info:
            info.update({'project_id': info.pop('tenantId')})
        if 'tenant_id' in info:
            info.update({'project_id': info.pop('tenant_id')})
        return zip(*sorted(info.items()))
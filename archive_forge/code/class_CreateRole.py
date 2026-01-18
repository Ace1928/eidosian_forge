import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateRole(command.ShowOne):
    _description = _('Create new role')

    def get_parser(self, prog_name):
        parser = super(CreateRole, self).get_parser(prog_name)
        parser.add_argument('role_name', metavar='<name>', help=_('New role name'))
        parser.add_argument('--or-show', action='store_true', help=_('Return existing role'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        try:
            role = identity_client.roles.create(parsed_args.role_name)
        except ks_exc.Conflict:
            if parsed_args.or_show:
                role = utils.find_resource(identity_client.roles, parsed_args.role_name)
                LOG.info(_('Returning existing role %s'), role.name)
            else:
                raise
        info = {}
        info.update(role._info)
        return zip(*sorted(info.items()))
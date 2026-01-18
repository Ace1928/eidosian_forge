import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteRole(command.Command):
    _description = _('Delete role(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteRole, self).get_parser(prog_name)
        parser.add_argument('roles', metavar='<role>', nargs='+', help=_('Role(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for role in parsed_args.roles:
            try:
                role_obj = utils.find_resource(identity_client.roles, role)
                identity_client.roles.delete(role_obj.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete role with name or ID '%(role)s': %(e)s"), {'role': role, 'e': e})
        if errors > 0:
            total = len(parsed_args.roles)
            msg = _('%(errors)s of %(total)s roles failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)
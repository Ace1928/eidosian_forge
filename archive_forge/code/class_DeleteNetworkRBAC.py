import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteNetworkRBAC(command.Command):
    _description = _('Delete network RBAC policy(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkRBAC, self).get_parser(prog_name)
        parser.add_argument('rbac_policy', metavar='<rbac-policy>', nargs='+', help=_('RBAC policy(s) to delete (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for rbac in parsed_args.rbac_policy:
            try:
                obj = client.find_rbac_policy(rbac, ignore_missing=False)
                client.delete_rbac_policy(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete RBAC policy with ID '%(rbac)s': %(e)s"), {'rbac': rbac, 'e': e})
        if result > 0:
            total = len(parsed_args.rbac_policy)
            msg = _('%(result)s of %(total)s RBAC policies failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
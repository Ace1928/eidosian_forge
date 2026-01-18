import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteConntrackHelper(command.Command):
    _description = _('Delete L3 conntrack helper')

    def get_parser(self, prog_name):
        parser = super(DeleteConntrackHelper, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router that the conntrack helper belong to'))
        parser.add_argument('conntrack_helper_id', metavar='<conntrack-helper-id>', nargs='+', help=_('The ID of the conntrack helper(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        router = client.find_router(parsed_args.router, ignore_missing=False)
        for ct_helper in parsed_args.conntrack_helper_id:
            try:
                client.delete_conntrack_helper(ct_helper, router.id, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete L3 conntrack helper with ID '%(ct_helper)s': %(e)s"), {'ct_helper': ct_helper, 'e': e})
        if result > 0:
            total = len(parsed_args.conntrack_helper_id)
            msg = _('%(result)s of %(total)s L3 conntrack helpers failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
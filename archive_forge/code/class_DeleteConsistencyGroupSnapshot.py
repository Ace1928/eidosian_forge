import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteConsistencyGroupSnapshot(command.Command):
    _description = _('Delete consistency group snapshot(s).')

    def get_parser(self, prog_name):
        parser = super(DeleteConsistencyGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('consistency_group_snapshot', metavar='<consistency-group-snapshot>', nargs='+', help=_('Consistency group snapshot(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for snapshot in parsed_args.consistency_group_snapshot:
            try:
                snapshot_id = utils.find_resource(volume_client.cgsnapshots, snapshot).id
                volume_client.cgsnapshots.delete(snapshot_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete consistency group snapshot with name or ID '%(snapshot)s': %(e)s") % {'snapshot': snapshot, 'e': e})
        if result > 0:
            total = len(parsed_args.consistency_group_snapshot)
            msg = _('%(result)s of %(total)s consistency group snapshots failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
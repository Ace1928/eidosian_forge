import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils
class DeleteShareReplica(command.Command):
    """Delete one or more share replicas."""
    _description = _('Delete one or more share replicas')

    def get_parser(self, prog_name):
        parser = super(DeleteShareReplica, self).get_parser(prog_name)
        parser.add_argument('replica', metavar='<replica>', nargs='+', help=_('Name or ID of the replica(s) to delete'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Attempt to force delete a replica on its backend. Using this option will purge the replica from Manila even if it is not cleaned up on the backend. '))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share replica deletion'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for replica in parsed_args.replica:
            try:
                replica_obj = osc_utils.find_resource(share_client.share_replicas, replica)
                share_client.share_replicas.delete(replica_obj, force=parsed_args.force)
                if parsed_args.wait:
                    if not osc_utils.wait_for_delete(manager=share_client.share_replicas, res_id=replica_obj.id):
                        result += 1
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete a share replica with name or ID '%(replica)s': %(e)s"), {'replica': replica, 'e': e})
        if result > 0:
            total = len(parsed_args.replica)
            msg = _('%(result)s of %(total)s replicas failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteConsistencyGroup(command.Command):
    _description = _('Delete consistency group(s).')

    def get_parser(self, prog_name):
        parser = super(DeleteConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('consistency_groups', metavar='<consistency-group>', nargs='+', help=_('Consistency group(s) to delete (name or ID)'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Allow delete in state other than error or available'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for i in parsed_args.consistency_groups:
            try:
                consistency_group_id = utils.find_resource(volume_client.consistencygroups, i).id
                volume_client.consistencygroups.delete(consistency_group_id, parsed_args.force)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete consistency group with name or ID '%(consistency_group)s':%(e)s") % {'consistency_group': i, 'e': e})
        if result > 0:
            total = len(parsed_args.consistency_groups)
            msg = _('%(result)s of %(total)s consistency groups failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)
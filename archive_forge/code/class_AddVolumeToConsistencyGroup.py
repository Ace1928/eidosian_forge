import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class AddVolumeToConsistencyGroup(command.Command):
    _description = _('Add volume(s) to consistency group')

    def get_parser(self, prog_name):
        parser = super(AddVolumeToConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('consistency_group', metavar='<consistency-group>', help=_('Consistency group to contain <volume> (name or ID)'))
        parser.add_argument('volumes', metavar='<volume>', nargs='+', help=_('Volume(s) to add to <consistency-group> (name or ID) (repeat option to add multiple volumes)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result, add_uuid = _find_volumes(parsed_args.volumes, volume_client)
        if result > 0:
            total = len(parsed_args.volumes)
            LOG.error(_('%(result)s of %(total)s volumes failed to add.') % {'result': result, 'total': total})
        if add_uuid:
            add_uuid = add_uuid.rstrip(',')
            consistency_group_id = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group).id
            volume_client.consistencygroups.update(consistency_group_id, add_volumes=add_uuid)
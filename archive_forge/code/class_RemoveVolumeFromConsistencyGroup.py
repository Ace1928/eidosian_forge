import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class RemoveVolumeFromConsistencyGroup(command.Command):
    _description = _('Remove volume(s) from consistency group')

    def get_parser(self, prog_name):
        parser = super(RemoveVolumeFromConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('consistency_group', metavar='<consistency-group>', help=_('Consistency group containing <volume> (name or ID)'))
        parser.add_argument('volumes', metavar='<volume>', nargs='+', help=_('Volume(s) to remove from <consistency-group> (name or ID) (repeat option to remove multiple volumes)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result, remove_uuid = _find_volumes(parsed_args.volumes, volume_client)
        if result > 0:
            total = len(parsed_args.volumes)
            LOG.error(_('%(result)s of %(total)s volumes failed to remove.') % {'result': result, 'total': total})
        if remove_uuid:
            remove_uuid = remove_uuid.rstrip(',')
            consistency_group_id = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group).id
            volume_client.consistencygroups.update(consistency_group_id, remove_volumes=remove_uuid)
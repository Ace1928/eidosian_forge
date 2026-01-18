import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalVolumeTarget(command.Command):
    """Set baremetal volume target properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('volume_target', metavar='<volume target>', help=_('UUID of the volume target.'))
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', help=_('UUID of the node that this volume target belongs to.'))
        parser.add_argument('--type', dest='volume_type', metavar='<volume type>', help=_("Type of the volume target, e.g. 'iscsi', 'fibre_channel'."))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action='append', help=_('Key/value property related to the type of this volume target. Can be specified multiple times.'))
        parser.add_argument('--boot-index', dest='boot_index', metavar='<boot index>', type=int, help=_('Boot index of the volume target.'))
        parser.add_argument('--volume-id', dest='volume_id', metavar='<volume id>', help=_('ID of the volume associated with this target.'))
        parser.add_argument('--extra', dest='extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.boot_index is not None and parsed_args.boot_index < 0:
            raise exc.CommandError(_('Expected non-negative --boot-index, got %s') % parsed_args.boot_index)
        properties = []
        if parsed_args.node_uuid:
            properties.extend(utils.args_array_to_patch('add', ['node_uuid=%s' % parsed_args.node_uuid]))
        if parsed_args.volume_type:
            properties.extend(utils.args_array_to_patch('add', ['volume_type=%s' % parsed_args.volume_type]))
        if parsed_args.boot_index:
            properties.extend(utils.args_array_to_patch('add', ['boot_index=%s' % parsed_args.boot_index]))
        if parsed_args.volume_id:
            properties.extend(utils.args_array_to_patch('add', ['volume_id=%s' % parsed_args.volume_id]))
        if parsed_args.properties:
            properties.extend(utils.args_array_to_patch('add', ['properties/' + x for x in parsed_args.properties]))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if properties:
            baremetal_client.volume_target.update(parsed_args.volume_target, properties)
        else:
            self.log.warning('Please specify what to set.')
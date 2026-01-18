import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class UnsetBaremetalVolumeTarget(command.Command):
    """Unset baremetal volume target properties."""
    log = logging.getLogger(__name__ + 'UnsetBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(UnsetBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('volume_target', metavar='<volume target>', help=_('UUID of the volume target.'))
        parser.add_argument('--extra', dest='extra', metavar='<key>', action='append', help=_('Extra to unset (repeat option to unset multiple extras)'))
        parser.add_argument('--property', dest='properties', metavar='<key>', action='append', help='Property to unset on this baremetal volume target (repeat option to unset multiple properties).')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('remove', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.properties:
            properties.extend(utils.args_array_to_patch('remove', ['properties/' + x for x in parsed_args.properties]))
        if properties:
            baremetal_client.volume_target.update(parsed_args.volume_target, properties)
        else:
            self.log.warning('Please specify what to unset.')
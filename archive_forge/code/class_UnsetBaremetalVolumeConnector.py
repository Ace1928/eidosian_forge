import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class UnsetBaremetalVolumeConnector(command.Command):
    """Unset baremetal volume connector properties."""
    log = logging.getLogger(__name__ + 'UnsetBaremetalVolumeConnector')

    def get_parser(self, prog_name):
        parser = super(UnsetBaremetalVolumeConnector, self).get_parser(prog_name)
        parser.add_argument('volume_connector', metavar='<volume connector>', help=_('UUID of the volume connector.'))
        parser.add_argument('--extra', dest='extra', metavar='<key>', action='append', help=_('Extra to unset (repeat option to unset multiple extras)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('remove', ['extra/' + x for x in parsed_args.extra]))
        if properties:
            baremetal_client.volume_connector.update(parsed_args.volume_connector, properties)
        else:
            self.log.warning('Please specify what to unset.')
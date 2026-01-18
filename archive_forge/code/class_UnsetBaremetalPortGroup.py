import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class UnsetBaremetalPortGroup(command.Command):
    """Unset baremetal port group properties."""
    log = logging.getLogger(__name__ + '.UnsetBaremetalPortGroup')

    def get_parser(self, prog_name):
        parser = super(UnsetBaremetalPortGroup, self).get_parser(prog_name)
        parser.add_argument('portgroup', metavar='<port group>', help=_('Name or UUID of the port group.'))
        parser.add_argument('--name', action='store_true', help=_('Unset the name of the port group.'))
        parser.add_argument('--address', action='store_true', help=_('Unset the address of the port group.'))
        parser.add_argument('--extra', metavar='<key>', action='append', help=_('Extra to unset on this baremetal port group (repeat option to unset multiple extras).'))
        parser.add_argument('--property', dest='properties', metavar='<key>', action='append', help=_('Property to unset on this baremetal port group (repeat option to unset multiple properties).'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.name:
            properties.extend(utils.args_array_to_patch('remove', ['name']))
        if parsed_args.address:
            properties.extend(utils.args_array_to_patch('remove', ['address']))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('remove', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.properties:
            properties.extend(utils.args_array_to_patch('remove', ['properties/' + x for x in parsed_args.properties]))
        if properties:
            baremetal_client.portgroup.update(parsed_args.portgroup, properties)
        else:
            self.log.warning('Please specify what to unset.')
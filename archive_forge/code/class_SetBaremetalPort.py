import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalPort(command.Command):
    """Set baremetal port properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalPort')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalPort, self).get_parser(prog_name)
        parser.add_argument('port', metavar='<port>', help=_('UUID of the port'))
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', help=_('Set UUID of the node that this port belongs to'))
        parser.add_argument('--address', metavar='<address>', dest='address', help=_('Set MAC address for this port'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra to set on this baremetal port (repeat option to set multiple extras)'))
        parser.add_argument('--port-group', metavar='<uuid>', dest='portgroup_uuid', help=_('Set UUID of the port group that this port belongs to.'))
        parser.add_argument('--local-link-connection', metavar='<key=value>', action='append', help=_("Key/value metadata describing local link connection information. Valid keys are 'switch_info', 'switch_id', 'port_id' and 'hostname'. The keys 'switch_id' and 'port_id' are required. In case of a Smart NIC port, the required keys are 'port_id' and 'hostname'. Argument can be specified multiple times."))
        pxe_enabled_group = parser.add_mutually_exclusive_group(required=False)
        pxe_enabled_group.add_argument('--pxe-enabled', dest='pxe_enabled', default=None, action='store_true', help=_('Indicates that this port should be used when PXE booting this node (default)'))
        pxe_enabled_group.add_argument('--pxe-disabled', dest='pxe_enabled', default=None, action='store_false', help=_('Indicates that this port should not be used when PXE booting this node'))
        parser.add_argument('--physical-network', metavar='<physical network>', dest='physical_network', help=_('Set the name of the physical network to which this port is connected.'))
        parser.add_argument('--is-smartnic', dest='is_smartnic', action='store_true', help=_('Set port to be Smart NIC port'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.node_uuid:
            node_uuid = ['node_uuid=%s' % parsed_args.node_uuid]
            properties.extend(utils.args_array_to_patch('add', node_uuid))
        if parsed_args.address:
            address = ['address=%s' % parsed_args.address]
            properties.extend(utils.args_array_to_patch('add', address))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.portgroup_uuid:
            portgroup_uuid = ['portgroup_uuid=%s' % parsed_args.portgroup_uuid]
            properties.extend(utils.args_array_to_patch('add', portgroup_uuid))
        if parsed_args.local_link_connection:
            properties.extend(utils.args_array_to_patch('add', ['local_link_connection/' + x for x in parsed_args.local_link_connection]))
        if parsed_args.pxe_enabled is not None:
            properties.extend(utils.args_array_to_patch('add', ['pxe_enabled=%s' % parsed_args.pxe_enabled]))
        if parsed_args.physical_network:
            physical_network = ['physical_network=%s' % parsed_args.physical_network]
            properties.extend(utils.args_array_to_patch('add', physical_network))
        if parsed_args.is_smartnic:
            is_smartnic = ['is_smartnic=%s' % parsed_args.is_smartnic]
            properties.extend(utils.args_array_to_patch('add', is_smartnic))
        if properties:
            baremetal_client.port.update(parsed_args.port, properties)
        else:
            self.log.warning('Please specify what to set.')
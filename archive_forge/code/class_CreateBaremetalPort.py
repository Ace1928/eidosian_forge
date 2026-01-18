import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalPort(command.ShowOne):
    """Create a new port"""
    log = logging.getLogger(__name__ + '.CreateBaremetalPort')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalPort, self).get_parser(prog_name)
        parser.add_argument('address', metavar='<address>', help=_('MAC address for this port.'))
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', required=True, help=_('UUID of the node that this port belongs to.'))
        parser.add_argument('--uuid', dest='uuid', metavar='<uuid>', help=_('UUID of the port.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Argument can be specified multiple times.'))
        parser.add_argument('--local-link-connection', metavar='<key=value>', action='append', help=_("Key/value metadata describing Local link connection information. Valid keys are 'switch_info', 'switch_id', 'port_id' and 'hostname'. The keys 'switch_id' and 'port_id' are required. In case of a Smart NIC port, the required keys are 'port_id' and 'hostname'. Argument can be specified multiple times."))
        parser.add_argument('-l', dest='local_link_connection_deprecated', metavar='<key=value>', action='append', help=_("DEPRECATED. Please use --local-link-connection instead. Key/value metadata describing Local link connection information. Valid keys are 'switch_info', 'switch_id', and 'port_id'. The keys 'switch_id' and 'port_id' are required. Can be specified multiple times."))
        parser.add_argument('--pxe-enabled', metavar='<boolean>', help=_('Indicates whether this Port should be used when PXE booting this Node.'))
        parser.add_argument('--port-group', dest='portgroup_uuid', metavar='<uuid>', help=_('UUID of the port group that this port belongs to.'))
        parser.add_argument('--physical-network', dest='physical_network', metavar='<physical network>', help=_('Name of the physical network to which this port is connected.'))
        parser.add_argument('--is-smartnic', dest='is_smartnic', action='store_true', help=_('Indicates whether this Port is a Smart NIC port'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.local_link_connection_deprecated:
            self.log.warning('Please use --local-link-connection instead of -l, as it is deprecated and will be removed in future releases.')
            if parsed_args.local_link_connection:
                parsed_args.local_link_connection.extend(parsed_args.local_link_connection_deprecated)
            else:
                parsed_args.local_link_connection = parsed_args.local_link_connection_deprecated
        field_list = ['address', 'uuid', 'extra', 'node_uuid', 'pxe_enabled', 'local_link_connection', 'portgroup_uuid', 'physical_network']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        fields = utils.args_array_to_dict(fields, 'extra')
        fields = utils.args_array_to_dict(fields, 'local_link_connection')
        if parsed_args.is_smartnic:
            fields['is_smartnic'] = parsed_args.is_smartnic
        port = baremetal_client.port.create(**fields)
        data = dict([(f, getattr(port, f, '')) for f in res_fields.PORT_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)
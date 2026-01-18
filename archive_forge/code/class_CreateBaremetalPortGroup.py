import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalPortGroup(command.ShowOne):
    """Create a new baremetal port group."""
    log = logging.getLogger(__name__ + '.CreateBaremetalPortGroup')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalPortGroup, self).get_parser(prog_name)
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', required=True, help=_('UUID of the node that this port group belongs to.'))
        parser.add_argument('--address', metavar='<mac-address>', help=_('MAC address for this port group.'))
        parser.add_argument('--name', dest='name', help=_('Name of the port group.'))
        parser.add_argument('--uuid', dest='uuid', help=_('UUID of the port group.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        parser.add_argument('--mode', help=_('Mode of the port group. For possible values, refer to https://www.kernel.org/doc/Documentation/networking/bonding.txt.'))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action='append', help=_("Key/value property related to this port group's configuration. Can be specified multiple times."))
        standalone_ports_group = parser.add_mutually_exclusive_group()
        standalone_ports_group.add_argument('--support-standalone-ports', action='store_true', help=_('Ports that are members of this port group can be used as stand-alone ports. (default)'))
        standalone_ports_group.add_argument('--unsupport-standalone-ports', action='store_true', help=_('Ports that are members of this port group cannot be used as stand-alone ports.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        field_list = ['node_uuid', 'address', 'name', 'uuid', 'extra', 'mode', 'properties']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        if parsed_args.support_standalone_ports:
            fields['standalone_ports_supported'] = True
        if parsed_args.unsupport_standalone_ports:
            fields['standalone_ports_supported'] = False
        fields = utils.args_array_to_dict(fields, 'extra')
        fields = utils.args_array_to_dict(fields, 'properties')
        portgroup = baremetal_client.portgroup.create(**fields)
        data = dict([(f, getattr(portgroup, f, '')) for f in res_fields.PORTGROUP_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)
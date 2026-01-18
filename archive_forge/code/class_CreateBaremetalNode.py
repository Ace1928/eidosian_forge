import argparse
import itertools
import json
import logging
import sys
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class CreateBaremetalNode(command.ShowOne):
    """Register a new node with the baremetal service"""
    log = logging.getLogger(__name__ + '.CreateBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('--chassis-uuid', dest='chassis_uuid', metavar='<chassis>', help=_('UUID of the chassis that this node belongs to.'))
        parser.add_argument('--driver', metavar='<driver>', required=True, help=_('Driver used to control the node [REQUIRED].'))
        parser.add_argument('--driver-info', metavar='<key=value>', action='append', help=_('Key/value pair used by the driver, such as out-of-band management credentials. Can be specified multiple times.'))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action='append', help=_('Key/value pair describing the physical characteristics of the node. This is exported to Nova and used by the scheduler. Can be specified multiple times.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        parser.add_argument('--uuid', metavar='<uuid>', help=_('Unique UUID for the node.'))
        parser.add_argument('--name', metavar='<name>', help=_('Unique name for the node.'))
        parser.add_argument('--bios-interface', metavar='<bios_interface>', help=_("BIOS interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--boot-interface', metavar='<boot_interface>', help=_("Boot interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--console-interface', metavar='<console_interface>', help=_("Console interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--deploy-interface', metavar='<deploy_interface>', help=_("Deploy interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--inspect-interface', metavar='<inspect_interface>', help=_("Inspect interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--management-interface', metavar='<management_interface>', help=_("Management interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--network-data', metavar='<network data>', dest='network_data', help=NETWORK_DATA_ARG_HELP)
        parser.add_argument('--network-interface', metavar='<network_interface>', help=_('Network interface used for switching node to cleaning/provisioning networks.'))
        parser.add_argument('--power-interface', metavar='<power_interface>', help=_("Power interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--raid-interface', metavar='<raid_interface>', help=_("RAID interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--rescue-interface', metavar='<rescue_interface>', help=_("Rescue interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--storage-interface', metavar='<storage_interface>', help=_("Storage interface used by the node's driver."))
        parser.add_argument('--vendor-interface', metavar='<vendor_interface>', help=_("Vendor interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        parser.add_argument('--resource-class', metavar='<resource_class>', help=_('Resource class for mapping nodes to Nova flavors'))
        parser.add_argument('--conductor-group', metavar='<conductor_group>', help=_('Conductor group the node will belong to'))
        clean = parser.add_mutually_exclusive_group()
        clean.add_argument('--automated-clean', action='store_true', default=None, help=_('Enable automated cleaning for the node'))
        clean.add_argument('--no-automated-clean', action='store_false', dest='automated_clean', default=None, help=_('Explicitly disable automated cleaning for the node'))
        parser.add_argument('--owner', metavar='<owner>', help=_('Owner of the node.'))
        parser.add_argument('--lessee', metavar='<lessee>', help=_('Lessee of the node.'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the node.'))
        parser.add_argument('--shard', metavar='<shard>', help=_('Shard for the node.'))
        parser.add_argument('--parent-node', metavar='<parent_node>', help=_('Parent node for the node being created.'))
        parser.add_argument('--firmware-interface', metavar='<firmware_interface>', help=_("Firmware interface used by the node's driver. This is only applicable when the specified --driver is a hardware type."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        field_list = ['automated_clean', 'chassis_uuid', 'driver', 'driver_info', 'properties', 'extra', 'uuid', 'name', 'conductor_group', 'owner', 'description', 'lessee', 'shard', 'resource_class', 'parent_node'] + ['%s_interface' % iface for iface in SUPPORTED_INTERFACES]
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and (not v is None)))
        fields = utils.args_array_to_dict(fields, 'driver_info')
        fields = utils.args_array_to_dict(fields, 'extra')
        if parsed_args.network_data:
            fields['network_data'] = utils.handle_json_arg(parsed_args.network_data, 'static network configuration')
        fields = utils.args_array_to_dict(fields, 'properties')
        node = baremetal_client.node.create(**fields)._info
        node.pop('links', None)
        node.pop('ports', None)
        node.pop('portgroups', None)
        node.pop('states', None)
        node.pop('volume', None)
        node.setdefault('chassis_uuid', '')
        return self.dict2columns(node)
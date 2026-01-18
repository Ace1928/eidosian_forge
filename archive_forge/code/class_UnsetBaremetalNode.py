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
class UnsetBaremetalNode(command.Command):
    """Unset baremetal properties"""
    log = logging.getLogger(__name__ + '.UnsetBaremetalNode')

    def get_parser(self, prog_name):
        parser = super(UnsetBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes."))
        parser.add_argument('--instance-uuid', action='store_true', default=False, help=_('Unset instance UUID on this baremetal node'))
        parser.add_argument('--name', action='store_true', help=_('Unset the name of the node'))
        parser.add_argument('--resource-class', dest='resource_class', action='store_true', help=_('Unset the resource class of the node'))
        parser.add_argument('--target-raid-config', action='store_true', help=_('Unset the target RAID configuration of the node'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Property to unset on this baremetal node (repeat option to unset multiple properties)'))
        parser.add_argument('--extra', metavar='<key>', action='append', help=_('Extra to unset on this baremetal node (repeat option to unset multiple extras)'))
        parser.add_argument('--driver-info', metavar='<key>', action='append', help=_('Driver information to unset on this baremetal node (repeat option to unset multiple items in driver information)'))
        parser.add_argument('--instance-info', metavar='<key>', action='append', help=_('Instance information to unset on this baremetal node (repeat option to unset multiple instance information)'))
        parser.add_argument('--chassis-uuid', dest='chassis_uuid', action='store_true', help=_('Unset chassis UUID on this baremetal node'))
        parser.add_argument('--bios-interface', dest='bios_interface', action='store_true', help=_('Unset BIOS interface on this baremetal node'))
        parser.add_argument('--boot-interface', dest='boot_interface', action='store_true', help=_('Unset boot interface on this baremetal node'))
        parser.add_argument('--console-interface', dest='console_interface', action='store_true', help=_('Unset console interface on this baremetal node'))
        parser.add_argument('--deploy-interface', dest='deploy_interface', action='store_true', help=_('Unset deploy interface on this baremetal node'))
        parser.add_argument('--firmware-interface', dest='firmware_interface', action='store_true', help=_('Unset firmware interface on this baremetal node'))
        parser.add_argument('--inspect-interface', dest='inspect_interface', action='store_true', help=_('Unset inspect interface on this baremetal node'))
        parser.add_argument('--network-data', action='store_true', help=_('Unset network data on this baremetal port.'))
        parser.add_argument('--management-interface', dest='management_interface', action='store_true', help=_('Unset management interface on this baremetal node'))
        parser.add_argument('--network-interface', dest='network_interface', action='store_true', help=_('Unset network interface on this baremetal node'))
        parser.add_argument('--power-interface', dest='power_interface', action='store_true', help=_('Unset power interface on this baremetal node'))
        parser.add_argument('--raid-interface', dest='raid_interface', action='store_true', help=_('Unset RAID interface on this baremetal node'))
        parser.add_argument('--rescue-interface', dest='rescue_interface', action='store_true', help=_('Unset rescue interface on this baremetal node'))
        parser.add_argument('--storage-interface', dest='storage_interface', action='store_true', help=_('Unset storage interface on this baremetal node'))
        parser.add_argument('--vendor-interface', dest='vendor_interface', action='store_true', help=_('Unset vendor interface on this baremetal node'))
        parser.add_argument('--conductor-group', action='store_true', help=_('Unset conductor group for this baremetal node (the default group will be used)'))
        parser.add_argument('--automated-clean', action='store_true', help=_('Unset automated clean option on this baremetal node (the value from configuration will be used)'))
        parser.add_argument('--protected', action='store_true', help=_('Unset the protected flag on the node'))
        parser.add_argument('--protected-reason', action='store_true', help=_('Unset the protected reason (gets unset automatically when protected is unset)'))
        parser.add_argument('--retired', action='store_true', help=_('Unset the retired flag on the node'))
        parser.add_argument('--retired-reason', action='store_true', help=_('Unset the retired reason (gets unset automatically when retired is unset)'))
        parser.add_argument('--owner', action='store_true', help=_('Unset the owner field of the node'))
        parser.add_argument('--lessee', action='store_true', help=_('Unset the lessee field of the node'))
        parser.add_argument('--description', action='store_true', help=_('Unset the description field of the node'))
        parser.add_argument('--shard', action='store_true', help=_('Unset the shard field of the node'))
        parser.add_argument('--parent-node', action='store_true', help=_('Unset the parent node field of the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.target_raid_config:
            for node in parsed_args.nodes:
                baremetal_client.node.set_target_raid_config(node, {})
        properties = []
        for field in ['instance_uuid', 'name', 'chassis_uuid', 'resource_class', 'conductor_group', 'automated_clean', 'bios_interface', 'boot_interface', 'console_interface', 'deploy_interface', 'firmware_interface', 'inspect_interface', 'management_interface', 'network_interface', 'power_interface', 'raid_interface', 'rescue_interface', 'storage_interface', 'vendor_interface', 'protected', 'protected_reason', 'retired', 'retired_reason', 'owner', 'lessee', 'description', 'shard', 'parent_node']:
            if getattr(parsed_args, field):
                properties.extend(utils.args_array_to_patch('remove', [field]))
        if parsed_args.property:
            properties.extend(utils.args_array_to_patch('remove', ['properties/' + x for x in parsed_args.property]))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('remove', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.driver_info:
            properties.extend(utils.args_array_to_patch('remove', ['driver_info/' + x for x in parsed_args.driver_info]))
        if parsed_args.instance_info:
            properties.extend(utils.args_array_to_patch('remove', ['instance_info/' + x for x in parsed_args.instance_info]))
        if parsed_args.network_data:
            properties.extend(utils.args_array_to_patch('remove', ['network_data']))
        if properties:
            for node in parsed_args.nodes:
                baremetal_client.node.update(node, properties)
        elif not parsed_args.target_raid_config:
            self.log.warning('Please specify what to unset.')
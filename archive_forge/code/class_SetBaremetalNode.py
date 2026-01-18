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
class SetBaremetalNode(command.Command):
    """Set baremetal properties"""
    log = logging.getLogger(__name__ + '.SetBaremetalNode')

    def _add_interface_args(self, parser, iface, set_help, reset_help):
        grp = parser.add_mutually_exclusive_group()
        grp.add_argument('--%s-interface' % iface, metavar='<%s_interface>' % iface, help=set_help)
        grp.add_argument('--reset-%s-interface' % iface, action='store_true', help=reset_help)

    def get_parser(self, prog_name):
        parser = super(SetBaremetalNode, self).get_parser(prog_name)
        parser.add_argument('nodes', metavar='<node>', nargs='+', help=_("Names or UUID's of the nodes."))
        parser.add_argument('--instance-uuid', metavar='<uuid>', help=_('Set instance UUID of node to <uuid>'))
        parser.add_argument('--name', metavar='<name>', help=_('Set the name of the node'))
        parser.add_argument('--chassis-uuid', metavar='<chassis UUID>', help=_('Set the chassis for the node'))
        parser.add_argument('--driver', metavar='<driver>', help=_('Set the driver for the node'))
        self._add_interface_args(parser, 'bios', set_help=_('Set the BIOS interface for the node'), reset_help=_('Reset the BIOS interface to its hardware type default'))
        self._add_interface_args(parser, 'boot', set_help=_('Set the boot interface for the node'), reset_help=_('Reset the boot interface to its hardware type default'))
        self._add_interface_args(parser, 'console', set_help=_('Set the console interface for the node'), reset_help=_('Reset the console interface to its hardware type default'))
        self._add_interface_args(parser, 'deploy', set_help=_('Set the deploy interface for the node'), reset_help=_('Reset the deploy interface to its hardware type default'))
        self._add_interface_args(parser, 'firmware', set_help=_('Set the firmware interface for the node'), reset_help=_('Reset the firmware interface for its hardware type default'))
        self._add_interface_args(parser, 'inspect', set_help=_('Set the inspect interface for the node'), reset_help=_('Reset the inspect interface to its hardware type default'))
        self._add_interface_args(parser, 'management', set_help=_('Set the management interface for the node'), reset_help=_('Reset the management interface to its hardware type default'))
        self._add_interface_args(parser, 'network', set_help=_('Set the network interface for the node'), reset_help=_('Reset the network interface to its hardware type default'))
        parser.add_argument('--network-data', metavar='<network data>', dest='network_data', help=NETWORK_DATA_ARG_HELP)
        self._add_interface_args(parser, 'power', set_help=_('Set the power interface for the node'), reset_help=_('Reset the power interface to its hardware type default'))
        self._add_interface_args(parser, 'raid', set_help=_('Set the RAID interface for the node'), reset_help=_('Reset the RAID interface to its hardware type default'))
        self._add_interface_args(parser, 'rescue', set_help=_('Set the rescue interface for the node'), reset_help=_('Reset the rescue interface to its hardware type default'))
        self._add_interface_args(parser, 'storage', set_help=_('Set the storage interface for the node'), reset_help=_('Reset the storage interface to its hardware type default'))
        self._add_interface_args(parser, 'vendor', set_help=_('Set the vendor interface for the node'), reset_help=_('Reset the vendor interface to its hardware type default'))
        parser.add_argument('--reset-interfaces', action='store_true', default=None, help=_('Reset all interfaces not specified explicitly to their default implementations. Only valid with --driver.'))
        parser.add_argument('--resource-class', metavar='<resource_class>', help=_('Set the resource class for the node'))
        parser.add_argument('--conductor-group', metavar='<conductor_group>', help=_('Set the conductor group for the node'))
        clean = parser.add_mutually_exclusive_group()
        clean.add_argument('--automated-clean', action='store_true', default=None, help=_('Enable automated cleaning for the node'))
        clean.add_argument('--no-automated-clean', action='store_false', dest='automated_clean', default=None, help=_('Explicitly disable automated cleaning for the node'))
        parser.add_argument('--protected', action='store_true', help=_('Mark the node as protected'))
        parser.add_argument('--protected-reason', metavar='<protected_reason>', help=_('Set the reason of marking the node as protected'))
        parser.add_argument('--retired', action='store_true', help=_('Mark the node as retired'))
        parser.add_argument('--retired-reason', metavar='<retired_reason>', help=_('Set the reason of marking the node as retired'))
        parser.add_argument('--target-raid-config', metavar='<target_raid_config>', help=_('Set the target RAID configuration (JSON) for the node. This can be one of: 1. a file containing YAML data of the RAID configuration; 2. "-" to read the contents from standard input; or 3. a valid JSON string.'))
        parser.add_argument('--property', metavar='<key=value>', action='append', help=_('Property to set on this baremetal node (repeat option to set multiple properties)'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra to set on this baremetal node (repeat option to set multiple extras)'))
        parser.add_argument('--driver-info', metavar='<key=value>', action='append', help=_('Driver information to set on this baremetal node (repeat option to set multiple driver infos)'))
        parser.add_argument('--instance-info', metavar='<key=value>', action='append', help=_('Instance information to set on this baremetal node (repeat option to set multiple instance infos)'))
        (parser.add_argument('--owner', metavar='<owner>', help=_('Set the owner for the node')),)
        (parser.add_argument('--lessee', metavar='<lessee>', help=_('Set the lessee for the node')),)
        parser.add_argument('--description', metavar='<description>', help=_('Set the description for the node'))
        parser.add_argument('--shard', metavar='<shard>', help=_('Set the shard for the node'))
        parser.add_argument('--parent-node', metavar='<parent_node>', help=_('Set the parent node for the node'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        if parsed_args.name and len(parsed_args.nodes) > 1:
            raise exc.CommandError(_('--name cannot be used with more than one node'))
        if parsed_args.instance_uuid and len(parsed_args.nodes) > 1:
            raise exc.CommandError(_('--instance-uuid cannot be used with more than one node'))
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.target_raid_config:
            raid_config = parsed_args.target_raid_config
            raid_config = utils.handle_json_arg(raid_config, 'target_raid_config')
            for node in parsed_args.nodes:
                baremetal_client.node.set_target_raid_config(node, raid_config)
        properties = []
        for field in ['instance_uuid', 'name', 'chassis_uuid', 'driver', 'resource_class', 'conductor_group', 'protected', 'protected_reason', 'retired', 'retired_reason', 'owner', 'lessee', 'description', 'shard', 'parent_node']:
            value = getattr(parsed_args, field)
            if value:
                properties.extend(utils.args_array_to_patch('add', ['%s=%s' % (field, value)]))
        if parsed_args.automated_clean is not None:
            properties.extend(utils.args_array_to_patch('add', ['automated_clean=%s' % parsed_args.automated_clean]))
        if parsed_args.reset_interfaces and (not parsed_args.driver):
            raise exc.CommandError(_('--reset-interfaces can only be specified with --driver'))
        for iface in SUPPORTED_INTERFACES:
            field = '%s_interface' % iface
            if getattr(parsed_args, field):
                properties.extend(utils.args_array_to_patch('add', ['%s_interface=%s' % (iface, getattr(parsed_args, field))]))
            elif getattr(parsed_args, 'reset_%s_interface' % iface):
                properties.extend(utils.args_array_to_patch('remove', ['%s_interface' % iface]))
        if parsed_args.property:
            properties.extend(utils.args_array_to_patch('add', ['properties/' + x for x in parsed_args.property]))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if parsed_args.driver_info:
            properties.extend(utils.args_array_to_patch('add', ['driver_info/' + x for x in parsed_args.driver_info]))
        if parsed_args.instance_info:
            properties.extend(utils.args_array_to_patch('add', ['instance_info/' + x for x in parsed_args.instance_info]))
        if parsed_args.network_data:
            network_data = utils.handle_json_arg(parsed_args.network_data, 'static network configuration')
            network_data = ['network_data=%s' % json.dumps(network_data)]
            properties.extend(utils.args_array_to_patch('add', network_data))
        if properties:
            for node in parsed_args.nodes:
                baremetal_client.node.update(node, properties, reset_interfaces=parsed_args.reset_interfaces)
        elif not parsed_args.target_raid_config:
            self.log.warning('Please specify what to set.')
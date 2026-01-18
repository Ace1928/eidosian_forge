import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateNetworkSegmentRange(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create new network segment range')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkSegmentRange, self).get_parser(prog_name)
        shared_group = parser.add_mutually_exclusive_group()
        shared_group.add_argument('--private', dest='private', action='store_true', help=_('Network segment range is assigned specifically to the project'))
        shared_group.add_argument('--shared', dest='shared', action='store_true', help=_('Network segment range is shared with other projects'))
        parser.add_argument('name', metavar='<name>', help=_('Name of new network segment range'))
        parser.add_argument('--project', metavar='<project>', help=_('Network segment range owner (name or ID). Optional when the segment range is shared'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--network-type', metavar='<network-type>', choices=['geneve', 'gre', 'vlan', 'vxlan'], required=True, help=_('Network type of this network segment range (geneve, gre, vlan or vxlan)'))
        parser.add_argument('--physical-network', metavar='<physical-network-name>', help=_('Physical network name of this network segment range'))
        parser.add_argument('--minimum', metavar='<minimum-segmentation-id>', type=int, required=True, help=_('Minimum segment identifier for this network segment range which is based on the network type, VLAN ID for vlan network type and tunnel ID for geneve, gre and vxlan network types'))
        parser.add_argument('--maximum', metavar='<maximum-segmentation-id>', type=int, required=True, help=_('Maximum segment identifier for this network segment range which is based on the network type, VLAN ID for vlan network type and tunnel ID for geneve, gre and vxlan network types'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            network_client.find_extension('network-segment-range', ignore_missing=False)
        except Exception as e:
            msg = _('Network segment range create not supported by Network API: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        identity_client = self.app.client_manager.identity
        if not parsed_args.private and parsed_args.project:
            msg = _('--project is only allowed with --private')
            raise exceptions.CommandError(msg)
        if parsed_args.network_type.lower() != 'vlan' and parsed_args.physical_network:
            msg = _('--physical-network is only allowed with --network-type vlan')
            raise exceptions.CommandError(msg)
        attrs = {}
        if parsed_args.shared or parsed_args.private:
            attrs['shared'] = parsed_args.shared
        else:
            attrs['shared'] = True
        attrs['network_type'] = parsed_args.network_type
        attrs['minimum'] = parsed_args.minimum
        attrs['maximum'] = parsed_args.maximum
        if parsed_args.name:
            attrs['name'] = parsed_args.name
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            if project_id:
                attrs['project_id'] = project_id
            else:
                msg = _('Failed to create the network segment range for project %(project_id)s') % parsed_args.project_id
                raise exceptions.CommandError(msg)
        elif not attrs['shared']:
            attrs['project_id'] = self.app.client_manager.auth_ref.project_id
        if parsed_args.physical_network:
            attrs['physical_network'] = parsed_args.physical_network
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = network_client.create_network_segment_range(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        data = _update_additional_fields_from_props(columns, props=data)
        return (display_columns, data)
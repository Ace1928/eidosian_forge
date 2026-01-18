import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class CreateShareNetwork(command.ShowOne):
    """Create a share network."""
    _description = _('Create a share network')

    def get_parser(self, prog_name):
        parser = super(CreateShareNetwork, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<share-network>', help=_('Add a name to the share network (Optional)'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Add a description to the share network (Optional).'))
        parser.add_argument('--neutron-net-id', metavar='<neutron-net-id>', default=None, help=_("ID of the neutron network that must be associated with the share network (Optional). The network specified will be associated with the 'default' share network subnet, unless 'availability-zone' is also specified."))
        parser.add_argument('--neutron-subnet-id', metavar='<neutron-subnet-id>', default=None, help=_("ID of the neutron sub-network that must be associated with the share network (Optional). The subnet specified will be associated with the 'default' share network subnet, unless 'availability-zone' is also specified."))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', default=None, help=_("Name or ID of the avalilability zone to assign the specified network subnet parameters to. Must be used in conjunction with 'neutron-net-id' and 'neutron-subnet-id'. Do not specify this parameter if the network must be available across all availability zones ('default' share network subnet). Available only for microversion >= 2.51."))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        availability_zone = None
        if parsed_args.availability_zone and share_client.api_version < api_versions.APIVersion('2.51'):
            raise exceptions.CommandError('Availability zone can be specified only with manila API version >= 2.51')
        elif parsed_args.availability_zone:
            availability_zone = parsed_args.availability_zone
        share_network = share_client.share_networks.create(name=parsed_args.name, description=parsed_args.description, neutron_net_id=parsed_args.neutron_net_id, neutron_subnet_id=parsed_args.neutron_subnet_id, availability_zone=availability_zone)
        share_network_data = share_network._info
        share_network_data.pop('links', None)
        if parsed_args.formatter == 'table':
            share_network_data['share_network_subnets'] = cliutils.convert_dict_list_to_string(share_network_data['share_network_subnets'])
        return self.dict2columns(share_network_data)
import logging
from operator import xor
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class UnsetShareNetworkSubnet(command.Command):
    """Unset a share network subnet property."""
    _description = _('Unset a share network subnet property')

    def get_parser(self, prog_name):
        parser = super(UnsetShareNetworkSubnet, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', help=_('Share network name or ID.'))
        parser.add_argument('share_network_subnet', metavar='<share-network-subnet>', help=_('ID of share network subnet to set a property.'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Remove a property from share network subnet (repeat option to remove multiple properties). Available only for microversion >= 2.78.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if parsed_args.property and share_client.api_version < api_versions.APIVersion('2.78'):
            raise exceptions.CommandError('Property can be specified only with manila API version >= 2.78.')
        share_network_id = oscutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        if parsed_args.property:
            result = 0
            for key in parsed_args.property:
                try:
                    share_client.share_network_subnets.delete_metadata(share_network_id, [key], subresource=parsed_args.share_network_subnet)
                except Exception as e:
                    result += 1
                    LOG.error("Failed to unset subnet property '%(key)s': %(e)s", {'key': key, 'e': e})
            if result > 0:
                total = len(parsed_args.property)
                raise exceptions.CommandError(f'{result} of {total} subnet properties failed to be unset.')
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateSubnetPool(neutronV20.UpdateCommand):
    """Update subnetpool's information."""
    resource = 'subnetpool'

    def add_known_arguments(self, parser):
        add_updatable_arguments(parser)
        parser.add_argument('--name', help=_('Updated name of the subnetpool.'))
        addrscope_args = parser.add_mutually_exclusive_group()
        addrscope_args.add_argument('--address-scope', metavar='ADDRSCOPE', help=_('ID or name of the address scope with which the subnetpool is associated. Prefixes must be unique across address scopes.'))
        addrscope_args.add_argument('--no-address-scope', action='store_true', help=_('Detach subnetpool from the address scope.'))

    def args2body(self, parsed_args):
        body = {}
        updatable_args2body(parsed_args, body)
        if parsed_args.no_address_scope:
            body['address_scope_id'] = None
        elif parsed_args.address_scope:
            _addrscope_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'address_scope', parsed_args.address_scope)
            body['address_scope_id'] = _addrscope_id
        return {'subnetpool': body}
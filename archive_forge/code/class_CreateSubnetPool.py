from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateSubnetPool(neutronV20.CreateCommand):
    """Create a subnetpool for a given tenant."""
    resource = 'subnetpool'

    def add_known_arguments(self, parser):
        add_updatable_arguments(parser, for_create=True)
        parser.add_argument('--shared', action='store_true', help=_('Set the subnetpool as shared.'))
        parser.add_argument('name', metavar='NAME', help=_('Name of the subnetpool to be created.'))
        parser.add_argument('--address-scope', metavar='ADDRSCOPE', help=_('ID or name of the address scope with which the subnetpool is associated. Prefixes must be unique across address scopes.'))

    def args2body(self, parsed_args):
        body = {'prefixes': parsed_args.prefixes}
        updatable_args2body(parsed_args, body)
        neutronV20.update_dict(parsed_args, body, ['tenant_id'])
        if parsed_args.shared:
            body['shared'] = True
        if parsed_args.address_scope:
            _addrscope_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'address_scope', parsed_args.address_scope)
            body['address_scope_id'] = _addrscope_id
        return {'subnetpool': body}
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateAddressScope(neutronV20.UpdateCommand):
    """Update an address scope."""
    resource = 'address_scope'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the address scope.'))
        utils.add_boolean_argument(parser, '--shared', help=_('Set sharing of address scope. (True means shared)'))

    def args2body(self, parsed_args):
        body = {}
        neutronV20.update_dict(parsed_args, body, ['name', 'shared'])
        return {self.resource: body}
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateRBACPolicy(neutronV20.UpdateCommand):
    """Update RBAC policy for given tenant."""
    resource = 'rbac_policy'
    allow_names = False

    def add_known_arguments(self, parser):
        parser.add_argument('--target-tenant', help=_('ID of the tenant to which the RBAC policy will be enforced.'))

    def args2body(self, parsed_args):
        body = {'target_tenant': parsed_args.target_tenant}
        return {self.resource: body}
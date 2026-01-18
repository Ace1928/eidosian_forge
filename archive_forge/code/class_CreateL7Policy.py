from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateL7Policy(neutronV20.CreateCommand):
    """LBaaS v2 Create L7 policy."""
    resource = 'l7policy'
    shadow_resource = 'lbaas_l7policy'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state_up', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--listener', required=True, metavar='LISTENER', help=_('ID or name of the listener this policy belongs to.'))

    def args2body(self, parsed_args):
        return _common_args2body(self.get_client(), parsed_args)
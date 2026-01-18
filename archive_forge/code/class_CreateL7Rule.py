from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateL7Rule(LbaasL7RuleMixin, neutronV20.CreateCommand):
    """LBaaS v2 Create L7 rule."""
    resource = 'rule'
    shadow_resource = 'lbaas_l7rule'

    def add_known_arguments(self, parser):
        super(CreateL7Rule, self).add_known_arguments(parser)
        _add_common_args(parser)
        parser.add_argument('--admin-state-down', dest='admin_state_up', action='store_false', help=_('Set admin state up to false'))

    def args2body(self, parsed_args):
        return _common_args2body(self.get_client(), parsed_args)
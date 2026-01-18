from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListL7Rule(LbaasL7RuleMixin, neutronV20.ListCommand):
    """LBaaS v2 List L7 rules that belong to a given L7 policy."""
    resource = 'rule'
    shadow_resource = 'lbaas_l7rule'
    pagination_support = True
    sorting_support = True
    list_columns = ['id', 'type', 'compare_type', 'invert', 'key', 'value', 'admin_state_up', 'status']

    def take_action(self, parsed_args):
        self.parent_id = _get_policy_id(self.get_client(), parsed_args.l7policy)
        self.values_specs.append('--l7policy_id=%s' % self.parent_id)
        return super(ListL7Rule, self).take_action(parsed_args)
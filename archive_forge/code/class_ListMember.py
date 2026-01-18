from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListMember(LbaasMemberMixin, neutronV20.ListCommand):
    """LBaaS v2 List members that belong to a given pool."""
    resource = 'member'
    shadow_resource = 'lbaas_member'
    list_columns = ['id', 'name', 'address', 'protocol_port', 'weight', 'subnet_id', 'admin_state_up', 'status']
    pagination_support = True
    sorting_support = True

    def take_action(self, parsed_args):
        self.parent_id = _get_pool_id(self.get_client(), parsed_args.pool)
        self.values_specs.append('--pool_id=%s' % self.parent_id)
        return super(ListMember, self).take_action(parsed_args)
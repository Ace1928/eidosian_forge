from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def _replace_dhcp_agents(self, dhcp_agent_ids):
    ret = self.client().list_dhcp_agent_hosting_networks(self.resource_id)
    old = set([agent['id'] for agent in ret['agents']])
    new = set(dhcp_agent_ids)
    for dhcp_agent_id in new - old:
        try:
            self.client().add_network_to_dhcp_agent(dhcp_agent_id, {'network_id': self.resource_id})
        except Exception as ex:
            if not self.client_plugin().is_conflict(ex):
                raise
    for dhcp_agent_id in old - new:
        try:
            self.client().remove_network_from_dhcp_agent(dhcp_agent_id, self.resource_id)
        except Exception as ex:
            if not (self.client_plugin().is_conflict(ex) or self.client_plugin().is_not_found(ex)):
                raise
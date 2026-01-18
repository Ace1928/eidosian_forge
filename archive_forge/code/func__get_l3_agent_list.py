from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def _get_l3_agent_list(self, props):
    l3_agent_id = props.pop(self.L3_AGENT_ID, None)
    l3_agent_ids = props.pop(self.L3_AGENT_IDS, None)
    if not l3_agent_ids and l3_agent_id:
        l3_agent_ids = [l3_agent_id]
    return l3_agent_ids
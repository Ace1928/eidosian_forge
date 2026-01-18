from neutron_lib.api.definitions import dhcpagentscheduler
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.definitions import test_agent as agent
class DHCPAgentSchedulerDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dhcpagentscheduler
    extension_resources = agent.AgentDefinitionTestCase.extension_resources
    extension_attributes = agent.AgentDefinitionTestCase.extension_attributes
    extension_subresources = (dhcpagentscheduler.DHCP_AGENTS, dhcpagentscheduler.DHCP_NETS)
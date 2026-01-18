from neutron_lib.api.definitions import agent_resources_synced
from neutron_lib.tests.unit.api.definitions import test_agent as base
class AgentResourcesSyncedDefinitionTestCase(base.AgentDefinitionTestCase):
    extension_module = agent_resources_synced
    extension_attributes = (agent_resources_synced.RESOURCES_SYNCED,)
from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
class FakeDRAgent(object):
    """Fake one or more dynamic routing agents."""

    @staticmethod
    def create_one_dragent(attrs=None):
        attrs = attrs or {}
        dragent_attrs = {'binary': 'neutron-bgp-dragent', 'admin_state_up': True, 'availability_zone': None, 'alive': True, 'topic': 'bgp_dragent', 'host': 'network-' + uuid.uuid4().hex, 'name': 'bgp-dragent-' + uuid.uuid4().hex, 'agent_type': 'BGP dynamic routing agent', 'id': uuid.uuid4().hex}
        dragent_attrs.update(attrs)
        return _agent.Agent(**dragent_attrs)

    @staticmethod
    def create_dragents(attrs=None, count=1):
        """Create one or multiple fake dynamic routing agents."""
        agents = []
        for i in range(count):
            agent = FakeDRAgent.create_one_dragent(attrs)
            agents.append(agent)
        return agents
from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
class TestRouterL3Agent(base.TestCase):

    def test_basic(self):
        sot = agent.RouterL3Agent()
        self.assertEqual('agent', sot.resource_key)
        self.assertEqual('agents', sot.resources_key)
        self.assertEqual('/routers/%(router_id)s/l3-agents', sot.base_path)
        self.assertEqual('l3-agent', sot.resource_name)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_retrieve)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)
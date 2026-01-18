from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import router
from openstack.tests.unit import base
class TestL3AgentRouters(base.TestCase):

    def test_basic(self):
        sot = router.L3AgentRouter()
        self.assertEqual('router', sot.resource_key)
        self.assertEqual('routers', sot.resources_key)
        self.assertEqual('/agents/%(agent_id)s/l3-routers', sot.base_path)
        self.assertEqual('l3-router', sot.resource_name)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_retrieve)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)
from unittest import mock
from openstack.compute.v2 import server_action
from openstack.tests.unit import base
class TestServerAction(base.TestCase):

    def setUp(self):
        super().setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.status_code = 200
        self.sess = mock.Mock()
        self.sess.post = mock.Mock(return_value=self.resp)

    def test_basic(self):
        sot = server_action.ServerAction()
        self.assertEqual('instanceAction', sot.resource_key)
        self.assertEqual('instanceActions', sot.resources_key)
        self.assertEqual('/servers/%(server_id)s/os-instance-actions', sot.base_path)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertDictEqual({'changes_before': 'changes-before', 'changes_since': 'changes-since', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = server_action.ServerAction(**EXAMPLE)
        self.assertEqual(EXAMPLE['action'], sot.action)
        self.assertEqual(EXAMPLE['message'], sot.message)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['request_id'], sot.request_id)
        self.assertEqual(EXAMPLE['start_time'], sot.start_time)
        self.assertEqual(EXAMPLE['user_id'], sot.user_id)
        self.assertEqual([server_action.ServerActionEvent(**e) for e in EXAMPLE['events']], sot.events)
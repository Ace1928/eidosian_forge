from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
@mock.patch.object(node.Node, '_translate_response', mock.Mock())
@mock.patch.object(node.Node, '_get_session', lambda self, x: x)
class TestNodeMaintenance(base.TestCase):

    def setUp(self):
        super(TestNodeMaintenance, self).setUp()
        self.node = node.Node.existing(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.1', retriable_status_codes=None)

    def test_set(self):
        self.node.set_maintenance(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': None}, headers=mock.ANY, microversion=mock.ANY)

    def test_set_with_reason(self):
        self.node.set_maintenance(self.session, 'No work on Monday')
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': 'No work on Monday'}, headers=mock.ANY, microversion=mock.ANY)

    def test_unset(self):
        self.node.unset_maintenance(self.session)
        self.session.delete.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json=None, headers=mock.ANY, microversion=mock.ANY)

    def test_set_via_update(self):
        self.node.is_maintenance = True
        self.node.commit(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': None}, headers=mock.ANY, microversion=mock.ANY)
        self.assertFalse(self.session.patch.called)

    def test_set_with_reason_via_update(self):
        self.node.is_maintenance = True
        self.node.maintenance_reason = 'No work on Monday'
        self.node.commit(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': 'No work on Monday'}, headers=mock.ANY, microversion=mock.ANY)
        self.assertFalse(self.session.patch.called)

    def test_set_with_other_fields(self):
        self.node.is_maintenance = True
        self.node.name = 'lazy-3000'
        self.node.commit(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': None}, headers=mock.ANY, microversion=mock.ANY)
        self.session.patch.assert_called_once_with('nodes/%s' % self.node.id, json=[{'path': '/name', 'op': 'replace', 'value': 'lazy-3000'}], headers=mock.ANY, microversion=mock.ANY)

    def test_set_with_reason_and_other_fields(self):
        self.node.is_maintenance = True
        self.node.maintenance_reason = 'No work on Monday'
        self.node.name = 'lazy-3000'
        self.node.commit(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': 'No work on Monday'}, headers=mock.ANY, microversion=mock.ANY)
        self.session.patch.assert_called_once_with('nodes/%s' % self.node.id, json=[{'path': '/name', 'op': 'replace', 'value': 'lazy-3000'}], headers=mock.ANY, microversion=mock.ANY)

    def test_no_reason_without_maintenance(self):
        self.node.maintenance_reason = 'Can I?'
        self.assertRaises(ValueError, self.node.commit, self.session)
        self.assertFalse(self.session.put.called)
        self.assertFalse(self.session.patch.called)

    def test_set_unset_maintenance(self):
        self.node.is_maintenance = True
        self.node.maintenance_reason = 'No work on Monday'
        self.node.commit(self.session)
        self.session.put.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json={'reason': 'No work on Monday'}, headers=mock.ANY, microversion=mock.ANY)
        self.node.is_maintenance = False
        self.node.commit(self.session)
        self.assertIsNone(self.node.maintenance_reason)
        self.session.delete.assert_called_once_with('nodes/%s/maintenance' % self.node.id, json=None, headers=mock.ANY, microversion=mock.ANY)
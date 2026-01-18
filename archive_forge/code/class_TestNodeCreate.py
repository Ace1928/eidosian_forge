from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(node.Node, '_translate_response', mock.Mock())
@mock.patch.object(node.Node, '_get_session', lambda self, x: x)
@mock.patch.object(node.Node, 'set_provision_state', autospec=True)
class TestNodeCreate(base.TestCase):

    def setUp(self):
        super(TestNodeCreate, self).setUp()
        self.new_state = None
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.1'
        self.node = node.Node(driver=FAKE['driver'])

        def _change_state(*args, **kwargs):
            self.node.provision_state = self.new_state
        self.session.post.side_effect = _change_state

    def test_available_old_version(self, mock_prov):
        self.node.provision_state = 'available'
        result = self.node.create(self.session)
        self.assertIs(result, self.node)
        self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion=self.session.default_microversion, params={})
        self.assertFalse(mock_prov.called)

    def test_available_new_version(self, mock_prov):
        self.session.default_microversion = '1.11'
        self.node.provision_state = 'available'
        result = self.node.create(self.session)
        self.assertIs(result, self.node)
        self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion='1.10', params={})
        mock_prov.assert_not_called()

    def test_no_enroll_in_old_version(self, mock_prov):
        self.node.provision_state = 'enroll'
        self.assertRaises(exceptions.NotSupported, self.node.create, self.session)
        self.assertFalse(self.session.post.called)
        self.assertFalse(mock_prov.called)

    def test_enroll_new_version(self, mock_prov):
        self.session.default_microversion = '1.11'
        self.node.provision_state = 'enroll'
        self.new_state = 'enroll'
        result = self.node.create(self.session)
        self.assertIs(result, self.node)
        self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion=self.session.default_microversion, params={})
        self.assertFalse(mock_prov.called)

    def test_no_manageable_in_old_version(self, mock_prov):
        self.node.provision_state = 'manageable'
        self.assertRaises(exceptions.NotSupported, self.node.create, self.session)
        self.assertFalse(self.session.post.called)
        self.assertFalse(mock_prov.called)

    def test_manageable_old_version(self, mock_prov):
        self.session.default_microversion = '1.4'
        self.node.provision_state = 'manageable'
        self.new_state = 'available'
        result = self.node.create(self.session)
        self.assertIs(result, self.node)
        self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion=self.session.default_microversion, params={})
        mock_prov.assert_called_once_with(self.node, self.session, 'manage', wait=True)

    def test_manageable_new_version(self, mock_prov):
        self.session.default_microversion = '1.11'
        self.node.provision_state = 'manageable'
        self.new_state = 'enroll'
        result = self.node.create(self.session)
        self.assertIs(result, self.node)
        self.session.post.assert_called_once_with(mock.ANY, json={'driver': FAKE['driver']}, headers=mock.ANY, microversion=self.session.default_microversion, params={})
        mock_prov.assert_called_once_with(self.node, self.session, 'manage', wait=True)
from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch('time.sleep', lambda _t: None)
@mock.patch.object(node.Node, 'fetch', autospec=True)
class TestNodeWaitForProvisionState(base.TestCase):

    def setUp(self):
        super(TestNodeWaitForProvisionState, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock()

    def test_success(self, mock_fetch):

        def _get_side_effect(_self, session):
            self.node.provision_state = 'manageable'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        node = self.node.wait_for_provision_state(self.session, 'manageable')
        self.assertIs(node, self.node)

    def test_failure(self, mock_fetch):

        def _get_side_effect(_self, session):
            self.node.provision_state = 'deploy failed'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        self.assertRaisesRegex(exceptions.ResourceFailure, 'failure state "deploy failed"', self.node.wait_for_provision_state, self.session, 'manageable')

    def test_failure_error(self, mock_fetch):

        def _get_side_effect(_self, session):
            self.node.provision_state = 'error'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        self.assertRaisesRegex(exceptions.ResourceFailure, 'failure state "error"', self.node.wait_for_provision_state, self.session, 'manageable')

    def test_enroll_as_failure(self, mock_fetch):

        def _get_side_effect(_self, session):
            self.node.provision_state = 'enroll'
            self.node.last_error = 'power failure'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        self.assertRaisesRegex(exceptions.ResourceFailure, 'failed to verify management credentials', self.node.wait_for_provision_state, self.session, 'manageable')

    def test_timeout(self, mock_fetch):
        self.assertRaises(exceptions.ResourceTimeout, self.node.wait_for_provision_state, self.session, 'manageable', timeout=0.001)

    def test_not_abort_on_failed_state(self, mock_fetch):

        def _get_side_effect(_self, session):
            self.node.provision_state = 'deploy failed'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        self.assertRaises(exceptions.ResourceTimeout, self.node.wait_for_provision_state, self.session, 'manageable', timeout=0.001, abort_on_failed_state=False)
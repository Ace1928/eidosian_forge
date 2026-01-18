from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import allocation
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('time.sleep', lambda _t: None)
@mock.patch.object(allocation.Allocation, 'fetch', autospec=True)
class TestWaitForAllocation(base.TestCase):

    def setUp(self):
        super(TestWaitForAllocation, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.52'
        self.session.log = mock.Mock()
        self.fake = dict(FAKE, state='allocating', node_uuid=None)
        self.allocation = allocation.Allocation(**self.fake)

    def test_already_active(self, mock_fetch):
        self.allocation.state = 'active'
        allocation = self.allocation.wait(None)
        self.assertIs(allocation, self.allocation)
        self.assertFalse(mock_fetch.called)

    def test_wait(self, mock_fetch):
        marker = [False]

        def _side_effect(allocation, session):
            if marker[0]:
                self.allocation.state = 'active'
                self.allocation.node_id = FAKE['node_uuid']
            else:
                marker[0] = True
        mock_fetch.side_effect = _side_effect
        allocation = self.allocation.wait(self.session)
        self.assertIs(allocation, self.allocation)
        self.assertEqual(2, mock_fetch.call_count)

    def test_failure(self, mock_fetch):
        marker = [False]

        def _side_effect(allocation, session):
            if marker[0]:
                self.allocation.state = 'error'
                self.allocation.last_error = 'boom!'
            else:
                marker[0] = True
        mock_fetch.side_effect = _side_effect
        self.assertRaises(exceptions.ResourceFailure, self.allocation.wait, self.session)
        self.assertEqual(2, mock_fetch.call_count)

    def test_failure_ignored(self, mock_fetch):
        marker = [False]

        def _side_effect(allocation, session):
            if marker[0]:
                self.allocation.state = 'error'
                self.allocation.last_error = 'boom!'
            else:
                marker[0] = True
        mock_fetch.side_effect = _side_effect
        allocation = self.allocation.wait(self.session, ignore_error=True)
        self.assertIs(allocation, self.allocation)
        self.assertEqual(2, mock_fetch.call_count)

    def test_timeout(self, mock_fetch):
        self.assertRaises(exceptions.ResourceTimeout, self.allocation.wait, self.session, timeout=0.001)
        mock_fetch.assert_called_with(self.allocation, self.session)
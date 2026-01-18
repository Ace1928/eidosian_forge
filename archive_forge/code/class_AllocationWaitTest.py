import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
@mock.patch('time.sleep', autospec=True)
@mock.patch('ironicclient.v1.allocation.AllocationManager.get', autospec=True)
class AllocationWaitTest(testtools.TestCase):

    def setUp(self):
        super(AllocationWaitTest, self).setUp()
        self.mgr = ironicclient.v1.allocation.AllocationManager(mock.Mock())

    def _fake_allocation(self, state, error=None):
        return mock.Mock(state=state, last_error=error)

    def test_success(self, mock_get, mock_sleep):
        allocations = [self._fake_allocation('allocating'), self._fake_allocation('allocating'), self._fake_allocation('active')]
        mock_get.side_effect = allocations
        result = self.mgr.wait('alloc1')
        self.assertIs(result, allocations[2])
        self.assertEqual(3, mock_get.call_count)
        self.assertEqual(2, mock_sleep.call_count)
        mock_get.assert_called_with(self.mgr, 'alloc1', os_ironic_api_version=None, global_request_id=None)

    def test_error(self, mock_get, mock_sleep):
        allocations = [self._fake_allocation('allocating'), self._fake_allocation('error')]
        mock_get.side_effect = allocations
        self.assertRaises(exc.StateTransitionFailed, self.mgr.wait, 'alloc1')
        self.assertEqual(2, mock_get.call_count)
        self.assertEqual(1, mock_sleep.call_count)
        mock_get.assert_called_with(self.mgr, 'alloc1', os_ironic_api_version=None, global_request_id=None)

    def test_timeout(self, mock_get, mock_sleep):
        mock_get.return_value = self._fake_allocation('allocating')
        self.assertRaises(exc.StateTransitionTimeout, self.mgr.wait, 'alloc1', timeout=0.001)
        mock_get.assert_called_with(self.mgr, 'alloc1', os_ironic_api_version=None, global_request_id=None)
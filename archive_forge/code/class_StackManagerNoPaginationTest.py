from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackManagerNoPaginationTest(testtools.TestCase):
    scenarios = [('total_0', dict(total=0)), ('total_1', dict(total=1)), ('total_9', dict(total=9)), ('total_10', dict(total=10)), ('total_11', dict(total=11)), ('total_19', dict(total=19)), ('total_20', dict(total=20)), ('total_21', dict(total=21)), ('total_49', dict(total=49)), ('total_50', dict(total=50)), ('total_51', dict(total=51)), ('total_95', dict(total=95))]
    limit = 50

    def mock_manager(self):
        manager = stacks.StackManager(None)
        manager._list = mock.MagicMock()

        def mock_list(*args, **kwargs):

            def results():
                for i in range(0, self.total):
                    stack_name = 'stack_%s' % (i + 1)
                    stack_id = 'abcd1234-%s' % (i + 1)
                    yield mock_stack(manager, stack_name, stack_id)
            return list(results())
        manager._list.side_effect = mock_list
        return manager

    def test_stack_list_no_pagination(self):
        manager = self.mock_manager()
        results = list(manager.list())
        manager._list.assert_called_once_with('/stacks?', 'stacks')
        self.assertEqual(self.total, len(results))
        if self.total > 0:
            self.assertEqual('stack_1', results[0].stack_name)
            self.assertEqual('stack_%s' % self.total, results[-1].stack_name)
from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackManagerPaginationTest(testtools.TestCase):
    scenarios = [('0_offset_0', dict(offset=0, total=0, results=((0, 0),))), ('1_offset_0', dict(offset=0, total=1, results=((0, 1),))), ('9_offset_0', dict(offset=0, total=9, results=((0, 9),))), ('10_offset_0', dict(offset=0, total=10, results=((0, 10), (10, 10)))), ('11_offset_0', dict(offset=0, total=11, results=((0, 10), (10, 11)))), ('11_offset_10', dict(offset=10, total=11, results=((10, 11),))), ('19_offset_10', dict(offset=10, total=19, results=((10, 19),))), ('20_offset_10', dict(offset=10, total=20, results=((10, 20), (20, 20)))), ('21_offset_10', dict(offset=10, total=21, results=((10, 20), (20, 21)))), ('21_offset_0', dict(offset=0, total=21, results=((0, 10), (10, 20), (20, 21)))), ('21_offset_20', dict(offset=20, total=21, results=((20, 21),))), ('95_offset_90', dict(offset=90, total=95, results=((90, 95),)))]
    limit = 50

    def mock_manager(self):
        manager = stacks.StackManager(None)
        manager._list = mock.MagicMock()

        def mock_list(arg_url, arg_response_key):
            try:
                result = self.results[self.result_index]
            except IndexError:
                return []
            self.result_index = self.result_index + 1
            limit_string = 'limit=%s' % self.limit
            self.assertIn(limit_string, arg_url)
            offset = result[0]
            if offset > 0:
                offset_string = 'marker=abcd1234-%s' % offset
                self.assertIn(offset_string, arg_url)

            def results():
                for i in range(*result):
                    self.limit -= 1
                    stack_name = 'stack_%s' % (i + 1)
                    stack_id = 'abcd1234-%s' % (i + 1)
                    yield mock_stack(manager, stack_name, stack_id)
            return list(results())
        manager._list.side_effect = mock_list
        return manager

    def test_stack_list_pagination(self):
        manager = self.mock_manager()
        list_params = {'limit': self.limit}
        if self.offset > 0:
            marker = 'abcd1234-%s' % self.offset
            list_params['marker'] = marker
        self.result_index = 0
        results = list(manager.list(**list_params))
        self.assertEqual(len(self.results), self.result_index)
        last_result = min(self.limit, self.total - self.offset)
        self.assertEqual(last_result, len(results))
        if last_result > 0:
            self.assertEqual('stack_%s' % (self.offset + 1), results[0].stack_name)
            self.assertEqual('stack_%s' % (self.offset + last_result), results[-1].stack_name)
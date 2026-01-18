import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
class DoActionOnManyTestCase(test_utils.TestCase):

    def _test_do_action_on_many(self, side_effect, fail):
        action = mock.Mock(side_effect=side_effect)
        if fail:
            self.assertRaises(exceptions.CommandError, utils.do_action_on_many, action, [1, 2], 'success with %s', 'error')
        else:
            utils.do_action_on_many(action, [1, 2], 'success with %s', 'error')
        action.assert_has_calls([mock.call(1), mock.call(2)])

    def test_do_action_on_many_success(self):
        self._test_do_action_on_many([None, None], fail=False)

    def test_do_action_on_many_first_fails(self):
        self._test_do_action_on_many([Exception(), None], fail=True)

    def test_do_action_on_many_last_fails(self):
        self._test_do_action_on_many([None, Exception()], fail=True)

    @mock.patch('sys.stdout', new_callable=io.StringIO)
    def _test_do_action_on_many_resource_string(self, resource, expected_string, mock_stdout):
        utils.do_action_on_many(mock.Mock(), [resource], 'success with %s', 'error')
        self.assertIn('success with %s' % expected_string, mock_stdout.getvalue())

    def test_do_action_on_many_resource_string_with_str(self):
        self._test_do_action_on_many_resource_string('resource1', 'resource1')

    def test_do_action_on_many_resource_string_with_human_id(self):
        resource = servers.Server(None, {'name': 'resource1'})
        self._test_do_action_on_many_resource_string(resource, 'resource1')

    def test_do_action_on_many_resource_string_with_id(self):
        resource = servers.Server(None, {'id': UUID})
        self._test_do_action_on_many_resource_string(resource, UUID)

    def test_do_action_on_many_resource_string_with_id_and_human_id(self):
        resource = servers.Server(None, {'name': 'resource1', 'id': UUID})
        self._test_do_action_on_many_resource_string(resource, 'resource1 (%s)' % UUID)
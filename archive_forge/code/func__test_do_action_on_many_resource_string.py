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
@mock.patch('sys.stdout', new_callable=io.StringIO)
def _test_do_action_on_many_resource_string(self, resource, expected_string, mock_stdout):
    utils.do_action_on_many(mock.Mock(), [resource], 'success with %s', 'error')
    self.assertIn('success with %s' % expected_string, mock_stdout.getvalue())
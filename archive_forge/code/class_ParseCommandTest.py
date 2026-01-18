import collections
from zunclient.common import cliutils
from zunclient.common import utils
from zunclient import exceptions as exc
from zunclient.tests.unit import utils as test_utils
class ParseCommandTest(test_utils.BaseTestCase):

    def test_no_command(self):
        command = []
        result = utils.parse_command(command)
        self.assertEqual('', result)

    def test_command_ls(self):
        command = ['ls', '-al']
        result = utils.parse_command(command)
        self.assertEqual('"ls" "-al"', result)

    def test_command_echo_hello(self):
        command = ['sh', '-c', 'echo hello']
        result = utils.parse_command(command)
        self.assertEqual('"sh" "-c" "echo hello"', result)
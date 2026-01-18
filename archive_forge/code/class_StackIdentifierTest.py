from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackIdentifierTest(testtools.TestCase):

    def test_stack_identifier(self):
        stack = mock_stack(None, 'the_stack', 'abcd1234')
        self.assertEqual('the_stack/abcd1234', stack.identifier)
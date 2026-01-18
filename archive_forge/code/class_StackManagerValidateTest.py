from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackManagerValidateTest(testtools.TestCase):

    def setUp(self):
        super(StackManagerValidateTest, self).setUp()
        self.mock_response = mock.MagicMock()
        self.mock_response.json.return_value = {'result': 'fake_response'}
        self.mock_response.headers = {'content-type': 'application/json'}
        self.mock_client = mock.MagicMock()
        self.mock_client.post.return_value = self.mock_response
        self.manager = stacks.StackManager(self.mock_client)

    def test_validate_show_nested(self):
        result = self.manager.validate(**{'show_nested': True})
        self.assertEqual(self.mock_response.json.return_value, result)
        self.mock_client.post.assert_called_once_with('/validate', params={'show_nested': True})

    def test_validate_show_nested_false(self):
        result = self.manager.validate(**{'show_nested': False})
        self.assertEqual(self.mock_response.json.return_value, result)
        self.mock_client.post.assert_called_once_with('/validate')

    def test_validate_show_nested_default(self):
        result = self.manager.validate()
        self.assertEqual(self.mock_response.json.return_value, result)
        self.mock_client.post.assert_called_once_with('/validate')

    def test_validate_ignore_errors(self):
        self.manager.validate(ignore_errors='99001,99002')
        self.mock_client.post.assert_called_once_with('/validate', params={'ignore_errors': '99001,99002'})
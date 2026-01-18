import builtins
from unittest import mock
import jsonschema
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import create_resources
class LoadFromFileTest(utils.BaseTestCase):

    @mock.patch.object(builtins, 'open', mock.mock_open(read_data='{"a": "b"}'))
    def test_load_json(self):
        fname = 'abc.json'
        res = create_resources.load_from_file(fname)
        self.assertEqual({'a': 'b'}, res)

    @mock.patch.object(builtins, 'open', mock.mock_open(read_data='{"a": "b"}'))
    def test_load_unknown_extension(self):
        fname = 'abc'
        self.assertRaisesRegex(exc.ClientException, 'must have .json or .yaml extension', create_resources.load_from_file, fname)

    @mock.patch.object(builtins, 'open', autospec=True)
    def test_load_ioerror(self, mock_open):
        mock_open.side_effect = IOError('file does not exist')
        fname = 'abc.json'
        self.assertRaisesRegex(exc.ClientException, 'Cannot read file', create_resources.load_from_file, fname)

    @mock.patch.object(builtins, 'open', mock.mock_open(read_data='{{bbb'))
    def test_load_incorrect_json(self):
        fname = 'abc.json'
        self.assertRaisesRegex(exc.ClientException, 'File "%s" is invalid' % fname, create_resources.load_from_file, fname)

    @mock.patch.object(builtins, 'open', mock.mock_open(read_data='---\na: b'))
    def test_load_yaml(self):
        fname = 'abc.yaml'
        res = create_resources.load_from_file(fname)
        self.assertEqual({'a': 'b'}, res)

    @mock.patch.object(builtins, 'open', mock.mock_open(read_data='---\na-: - b'))
    def test_load_incorrect_yaml(self):
        fname = 'abc.yaml'
        self.assertRaisesRegex(exc.ClientException, 'File "%s" is invalid' % fname, create_resources.load_from_file, fname)
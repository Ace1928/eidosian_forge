import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
class ArgsArrayToPatchTest(test_utils.BaseTestCase):

    def test_args_array_to_patch(self):
        my_args = {'attributes': ['str=foo', 'int=1', 'bool=true', 'list=[1, 2, 3]', 'dict={"foo": "bar"}'], 'op': 'add'}
        patch = utils.args_array_to_patch(my_args['op'], my_args['attributes'])
        self.assertEqual([{'op': 'add', 'value': 'foo', 'path': '/str'}, {'op': 'add', 'value': 1, 'path': '/int'}, {'op': 'add', 'value': True, 'path': '/bool'}, {'op': 'add', 'value': [1, 2, 3], 'path': '/list'}, {'op': 'add', 'value': {'foo': 'bar'}, 'path': '/dict'}], patch)

    def test_args_array_to_patch_format_error(self):
        my_args = {'attributes': ['foobar'], 'op': 'add'}
        self.assertRaises(exc.CommandError, utils.args_array_to_patch, my_args['op'], my_args['attributes'])

    def test_args_array_to_patch_remove(self):
        my_args = {'attributes': ['/foo', 'extra/bar'], 'op': 'remove'}
        patch = utils.args_array_to_patch(my_args['op'], my_args['attributes'])
        self.assertEqual([{'op': 'remove', 'path': '/foo'}, {'op': 'remove', 'path': '/extra/bar'}], patch)

    def test_args_array_to_patch_invalid_op(self):
        my_args = {'attributes': ['/foo', 'extra/bar'], 'op': 'invalid'}
        self.assertRaises(exc.CommandError, utils.args_array_to_patch, my_args['op'], my_args['attributes'])
import builtins
import collections
from unittest import mock
from oslo_serialization import jsonutils
import tempfile
from magnumclient.common import cliutils
from magnumclient.common import utils
from magnumclient import exceptions as exc
from magnumclient.tests import utils as test_utils
class SplitAndDeserializeTest(test_utils.BaseTestCase):

    def test_split_and_deserialize(self):
        ret = utils.split_and_deserialize('str=foo')
        self.assertEqual(('str', 'foo'), ret)
        ret = utils.split_and_deserialize('int=1')
        self.assertEqual(('int', 1), ret)
        ret = utils.split_and_deserialize('bool=false')
        self.assertEqual(('bool', False), ret)
        ret = utils.split_and_deserialize('list=[1, "foo", 2]')
        self.assertEqual(('list', [1, 'foo', 2]), ret)
        ret = utils.split_and_deserialize('dict={"foo": 1}')
        self.assertEqual(('dict', {'foo': 1}), ret)
        ret = utils.split_and_deserialize('str_int="1"')
        self.assertEqual(('str_int', '1'), ret)

    def test_split_and_deserialize_fail(self):
        self.assertRaises(exc.CommandError, utils.split_and_deserialize, 'foo:bar')
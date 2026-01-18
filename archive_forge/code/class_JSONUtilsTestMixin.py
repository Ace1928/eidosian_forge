import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
class JSONUtilsTestMixin(object):
    json_impl = None

    def setUp(self):
        super(JSONUtilsTestMixin, self).setUp()
        self.json_patcher = mock.patch.multiple(jsonutils, json=self.json_impl)
        self.json_impl_mock = self.json_patcher.start()

    def tearDown(self):
        self.json_patcher.stop()
        super(JSONUtilsTestMixin, self).tearDown()

    def test_dumps(self):
        self.assertEqual('{"a": "b"}', jsonutils.dumps({'a': 'b'}))

    def test_dumps_default(self):
        args = [ReprObject()]
        convert = functools.partial(jsonutils.to_primitive, fallback=repr)
        self.assertEqual('["repr"]', jsonutils.dumps(args, default=convert))

    def test_dump_as_bytes(self):
        self.assertEqual(b'{"a": "b"}', jsonutils.dump_as_bytes({'a': 'b'}))

    def test_dumps_namedtuple(self):
        n = collections.namedtuple('foo', 'bar baz')(1, 2)
        self.assertEqual('[1, 2]', jsonutils.dumps(n))

    def test_dump(self):
        expected = '{"a": "b"}'
        json_dict = {'a': 'b'}
        fp = io.StringIO()
        jsonutils.dump(json_dict, fp)
        self.assertEqual(expected, fp.getvalue())

    def test_dump_namedtuple(self):
        expected = '[1, 2]'
        json_dict = collections.namedtuple('foo', 'bar baz')(1, 2)
        fp = io.StringIO()
        jsonutils.dump(json_dict, fp)
        self.assertEqual(expected, fp.getvalue())

    def test_loads(self):
        self.assertEqual({'a': 'b'}, jsonutils.loads('{"a": "b"}'))

    def test_loads_unicode(self):
        self.assertIsInstance(jsonutils.loads(b'"foo"'), str)
        self.assertIsInstance(jsonutils.loads('"foo"'), str)
        i18n_str_unicode = '"тест"'
        self.assertIsInstance(jsonutils.loads(i18n_str_unicode), str)
        i18n_str = i18n_str_unicode.encode('utf-8')
        self.assertIsInstance(jsonutils.loads(i18n_str), str)

    def test_loads_with_kwargs(self):
        jsontext = '{"foo": 3}'
        result = jsonutils.loads(jsontext, parse_int=lambda x: 5)
        self.assertEqual(5, result['foo'])

    def test_load(self):
        jsontext = '{"a": "тэст"}'
        expected = {'a': 'тэст'}
        for encoding in ('utf-8', 'cp1251'):
            fp = io.BytesIO(jsontext.encode(encoding))
            result = jsonutils.load(fp, encoding=encoding)
            self.assertEqual(expected, result)
            for key, val in result.items():
                self.assertIsInstance(key, str)
                self.assertIsInstance(val, str)

    def test_dumps_exception_value(self):
        self.assertIn(jsonutils.dumps({'a': ValueError('hello')}), ['{"a": "ValueError(\'hello\',)"}', '{"a": "ValueError(\'hello\')"}'])
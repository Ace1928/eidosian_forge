from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
class ResourceManagerTest(testtools.TestCase):

    def _base_test(self, func, fields, expect, key):
        manager = resources.ResourceManager(FakeAPI(expect, key))
        with mock.patch.object(manager, '_resolve_stack_id') as mock_rslv, mock.patch.object(utils, 'get_response_body') as mock_resp:
            mock_resp.return_value = {key: key and {key: []} or {}}
            mock_rslv.return_value = 'teststack/abcd1234'
            getattr(manager, func)(**fields)
            mock_resp.assert_called_once_with(mock.ANY)
            mock_rslv.assert_called_once_with('teststack')

    def test_get(self):
        fields = {'stack_id': 'teststack', 'resource_name': 'testresource'}
        expect = ('GET', '/stacks/teststack/abcd1234/resources/testresource')
        key = 'resource'
        self._base_test('get', fields, expect, key)

    def test_get_with_attr(self):
        fields = {'stack_id': 'teststack', 'resource_name': 'testresource', 'with_attr': ['attr_a', 'attr_b']}
        expect = ('GET', '/stacks/teststack/abcd1234/resources/testresource?with_attr=attr_a&with_attr=attr_b')
        key = 'resource'
        self._base_test('get', fields, expect, key)

    def test_get_with_unicode_resource_name(self):
        fields = {'stack_id': 'teststack', 'resource_name': '工作'}
        expect = ('GET', '/stacks/teststack/abcd1234/resources/%E5%B7%A5%E4%BD%9C')
        key = 'resource'
        self._base_test('get', fields, expect, key)

    def _test_list(self, fields, expect):
        key = 'resources'

        class FakeResponse(object):

            def json(self):
                return {key: {}}

        class FakeClient(object):

            def get(self, *args, **kwargs):
                assert args[0] == expect
                return FakeResponse()
        manager = resources.ResourceManager(FakeClient())
        manager.list(**fields)

    def test_list(self):
        self._test_list(fields={'stack_id': 'teststack'}, expect='/stacks/teststack/resources')

    def test_list_nested(self):
        self._test_list(fields={'stack_id': 'teststack', 'nested_depth': '99'}, expect='/stacks/teststack/resources?%s' % parse.urlencode({'nested_depth': 99}, True))

    def test_list_filtering(self):
        self._test_list(fields={'stack_id': 'teststack', 'filters': {'name': 'rsc_1'}}, expect='/stacks/teststack/resources?%s' % parse.urlencode({'name': 'rsc_1'}, True))

    def test_list_detail(self):
        self._test_list(fields={'stack_id': 'teststack', 'with_detail': 'True'}, expect='/stacks/teststack/resources?%s' % parse.urlencode({'with_detail': True}, True))

    def test_metadata(self):
        fields = {'stack_id': 'teststack', 'resource_name': 'testresource'}
        expect = ('GET', '/stacks/teststack/abcd1234/resources/testresource/metadata')
        key = 'metadata'
        self._base_test('metadata', fields, expect, key)

    def test_generate_template(self):
        fields = {'resource_name': 'testresource'}
        expect = ('GET', '/resource_types/testresource/template')
        key = None

        class FakeAPI(object):
            """Fake API and ensure request url is correct."""

            def get(self, *args, **kwargs):
                assert ('GET', args[0]) == expect

            def json_request(self, *args, **kwargs):
                assert args == expect
                ret = key and {key: []} or {}
                return ({}, {key: ret})
        manager = resources.ResourceManager(FakeAPI())
        with mock.patch.object(utils, 'get_response_body') as mock_resp:
            mock_resp.return_value = {key: key and {key: []} or {}}
            manager.generate_template(**fields)
            mock_resp.assert_called_once_with(mock.ANY)

    def test_signal(self):
        fields = {'stack_id': 'teststack', 'resource_name': 'testresource', 'data': 'Some content'}
        expect = ('POST', '/stacks/teststack/abcd1234/resources/testresource/signal')
        key = 'signal'
        self._base_test('signal', fields, expect, key)

    def test_mark_unhealthy(self):
        fields = {'stack_id': 'teststack', 'resource_name': 'testresource', 'mark_unhealthy': 'True', 'resource_status_reason': 'Anything'}
        expect = ('PATCH', '/stacks/teststack/abcd1234/resources/testresource')
        key = 'mark_unhealthy'
        self._base_test('mark_unhealthy', fields, expect, key)
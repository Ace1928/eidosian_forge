import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestResourceActions(base.TestCase):

    def setUp(self):
        super(TestResourceActions, self).setUp()
        self.service_name = 'service'
        self.base_path = 'base_path'

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            resources_key = 'resources'
            allow_create = True
            allow_fetch = True
            allow_head = True
            allow_commit = True
            allow_delete = True
            allow_list = True
        self.test_class = Test
        self.request = mock.Mock(spec=resource._Request)
        self.request.url = 'uri'
        self.request.body = 'body'
        self.request.headers = 'headers'
        self.response = FakeResponse({})
        self.sot = Test(id='id')
        self.sot._prepare_request = mock.Mock(return_value=self.request)
        self.sot._translate_response = mock.Mock()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.create = mock.Mock(return_value=self.response)
        self.session.get = mock.Mock(return_value=self.response)
        self.session.put = mock.Mock(return_value=self.response)
        self.session.patch = mock.Mock(return_value=self.response)
        self.session.post = mock.Mock(return_value=self.response)
        self.session.delete = mock.Mock(return_value=self.response)
        self.session.head = mock.Mock(return_value=self.response)
        self.session.session = self.session
        self.session._get_connection = mock.Mock(return_value=self.cloud)
        self.session.default_microversion = None
        self.session.retriable_status_codes = None
        self.endpoint_data = mock.Mock(max_microversion='1.99', min_microversion=None)
        self.session.get_endpoint_data.return_value = self.endpoint_data

    def _test_create(self, cls, requires_id=False, prepend_key=False, microversion=None, base_path=None, params=None, id_marked_dirty=True, explicit_microversion=None, resource_request_key=None, resource_response_key=None):
        id = 'id' if requires_id else None
        sot = cls(id=id)
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        params = params or {}
        kwargs = params.copy()
        if explicit_microversion is not None:
            kwargs['microversion'] = explicit_microversion
            microversion = explicit_microversion
        result = sot.create(self.session, prepend_key=prepend_key, base_path=base_path, resource_request_key=resource_request_key, resource_response_key=resource_response_key, **kwargs)
        id_is_dirty = 'id' in sot._body._dirty
        self.assertEqual(id_marked_dirty, id_is_dirty)
        prepare_kwargs = {}
        if resource_request_key is not None:
            prepare_kwargs['resource_request_key'] = resource_request_key
        sot._prepare_request.assert_called_once_with(requires_id=requires_id, prepend_key=prepend_key, base_path=base_path, **prepare_kwargs)
        if requires_id:
            self.session.put.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, params=params)
        else:
            self.session.post.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, params=params)
        self.assertEqual(sot.microversion, microversion)
        res_kwargs = {}
        if resource_response_key is not None:
            res_kwargs['resource_response_key'] = resource_response_key
        sot._translate_response.assert_called_once_with(self.response, has_body=sot.has_body, **res_kwargs)
        self.assertEqual(result, sot)

    def test_put_create(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'PUT'
        self._test_create(Test, requires_id=True, prepend_key=True)

    def test_put_create_exclude_id(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'PUT'
            create_exclude_id_from_body = True
        self._test_create(Test, requires_id=True, prepend_key=True, id_marked_dirty=False)

    def test_put_create_with_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'PUT'
            _max_microversion = '1.42'
        self._test_create(Test, requires_id=True, prepend_key=True, microversion='1.42')

    def test_put_create_with_explicit_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'PUT'
            _max_microversion = '1.99'
        self._test_create(Test, requires_id=True, prepend_key=True, explicit_microversion='1.42')

    def test_put_create_with_params(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'PUT'
        self._test_create(Test, requires_id=True, prepend_key=True, params={'answer': 42})

    def test_post_create(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
        self._test_create(Test, requires_id=False, prepend_key=True)

    def test_post_create_override_request_key(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
            resource_key = 'SomeKey'
        self._test_create(Test, requires_id=False, prepend_key=True, resource_request_key='OtherKey')

    def test_post_create_override_response_key(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
            resource_key = 'SomeKey'
        self._test_create(Test, requires_id=False, prepend_key=True, resource_response_key='OtherKey')

    def test_post_create_override_key_both(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
            resource_key = 'SomeKey'
        self._test_create(Test, requires_id=False, prepend_key=True, resource_request_key='OtherKey', resource_response_key='SomeOtherKey')

    def test_post_create_base_path(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
        self._test_create(Test, requires_id=False, prepend_key=True, base_path='dummy')

    def test_post_create_with_params(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_create = True
            create_method = 'POST'
        self._test_create(Test, requires_id=False, prepend_key=True, params={'answer': 42})

    def test_fetch(self):
        result = self.sot.fetch(self.session)
        self.sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion=None, params={}, skip_cache=False)
        self.assertIsNone(self.sot.microversion)
        self.sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, self.sot)

    def test_fetch_with_override_key(self):
        result = self.sot.fetch(self.session, resource_response_key='SomeKey')
        self.sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion=None, params={}, skip_cache=False)
        self.assertIsNone(self.sot.microversion)
        self.sot._translate_response.assert_called_once_with(self.response, resource_response_key='SomeKey')
        self.assertEqual(result, self.sot)

    def test_fetch_with_params(self):
        result = self.sot.fetch(self.session, fields='a,b')
        self.sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion=None, params={'fields': 'a,b'}, skip_cache=False)
        self.assertIsNone(self.sot.microversion)
        self.sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, self.sot)

    def test_fetch_with_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_fetch = True
            _max_microversion = '1.42'
        sot = Test(id='id')
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        result = sot.fetch(self.session)
        sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion='1.42', params={}, skip_cache=False)
        self.assertEqual(sot.microversion, '1.42')
        sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, sot)

    def test_fetch_with_explicit_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_fetch = True
            _max_microversion = '1.99'
        sot = Test(id='id')
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        result = sot.fetch(self.session, microversion='1.42')
        sot._prepare_request.assert_called_once_with(requires_id=True, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion='1.42', params={}, skip_cache=False)
        self.assertEqual(sot.microversion, '1.42')
        sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, sot)

    def test_fetch_not_requires_id(self):
        result = self.sot.fetch(self.session, False)
        self.sot._prepare_request.assert_called_once_with(requires_id=False, base_path=None)
        self.session.get.assert_called_once_with(self.request.url, microversion=None, params={}, skip_cache=False)
        self.sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, self.sot)

    def test_fetch_base_path(self):
        result = self.sot.fetch(self.session, False, base_path='dummy')
        self.sot._prepare_request.assert_called_once_with(requires_id=False, base_path='dummy')
        self.session.get.assert_called_once_with(self.request.url, microversion=None, params={}, skip_cache=False)
        self.sot._translate_response.assert_called_once_with(self.response)
        self.assertEqual(result, self.sot)

    def test_head(self):
        result = self.sot.head(self.session)
        self.sot._prepare_request.assert_called_once_with(base_path=None)
        self.session.head.assert_called_once_with(self.request.url, microversion=None)
        self.assertIsNone(self.sot.microversion)
        self.sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, self.sot)

    def test_head_base_path(self):
        result = self.sot.head(self.session, base_path='dummy')
        self.sot._prepare_request.assert_called_once_with(base_path='dummy')
        self.session.head.assert_called_once_with(self.request.url, microversion=None)
        self.assertIsNone(self.sot.microversion)
        self.sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, self.sot)

    def test_head_with_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_head = True
            _max_microversion = '1.42'
        sot = Test(id='id')
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        result = sot.head(self.session)
        sot._prepare_request.assert_called_once_with(base_path=None)
        self.session.head.assert_called_once_with(self.request.url, microversion='1.42')
        self.assertEqual(sot.microversion, '1.42')
        sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, sot)

    def _test_commit(self, commit_method='PUT', prepend_key=True, has_body=True, microversion=None, commit_args=None, expected_args=None, base_path=None, explicit_microversion=None):
        self.sot.commit_method = commit_method
        self.sot._body = mock.Mock()
        self.sot._body.dirty = mock.Mock(return_value={'x': 'y'})
        commit_args = commit_args or {}
        if explicit_microversion is not None:
            commit_args['microversion'] = explicit_microversion
            microversion = explicit_microversion
        self.sot.commit(self.session, prepend_key=prepend_key, has_body=has_body, base_path=base_path, **commit_args)
        self.sot._prepare_request.assert_called_once_with(prepend_key=prepend_key, base_path=base_path)
        if commit_method == 'PATCH':
            self.session.patch.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
        elif commit_method == 'POST':
            self.session.post.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
        elif commit_method == 'PUT':
            self.session.put.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
        self.assertEqual(self.sot.microversion, microversion)
        self.sot._translate_response.assert_called_once_with(self.response, has_body=has_body)

    def test_commit_put(self):
        self._test_commit(commit_method='PUT', prepend_key=True, has_body=True)

    def test_commit_patch(self):
        self._test_commit(commit_method='PATCH', prepend_key=False, has_body=False)

    def test_commit_base_path(self):
        self._test_commit(commit_method='PUT', prepend_key=True, has_body=True, base_path='dummy')

    def test_commit_patch_retry_on_conflict(self):
        self._test_commit(commit_method='PATCH', commit_args={'retry_on_conflict': True}, expected_args={'retriable_status_codes': {409}})

    def test_commit_put_retry_on_conflict(self):
        self._test_commit(commit_method='PUT', commit_args={'retry_on_conflict': True}, expected_args={'retriable_status_codes': {409}})

    def test_commit_patch_no_retry_on_conflict(self):
        self.session.retriable_status_codes = {409, 503}
        self._test_commit(commit_method='PATCH', commit_args={'retry_on_conflict': False}, expected_args={'retriable_status_codes': {503}})

    def test_commit_put_no_retry_on_conflict(self):
        self.session.retriable_status_codes = {409, 503}
        self._test_commit(commit_method='PATCH', commit_args={'retry_on_conflict': False}, expected_args={'retriable_status_codes': {503}})

    def test_commit_put_explicit_microversion(self):
        self._test_commit(commit_method='PUT', prepend_key=True, has_body=True, explicit_microversion='1.42')

    def test_commit_not_dirty(self):
        self.sot._body = mock.Mock()
        self.sot._body.dirty = dict()
        self.sot._header = mock.Mock()
        self.sot._header.dirty = dict()
        self.sot.commit(self.session)
        self.session.put.assert_not_called()

    def test_patch_with_sdk_names(self):

        class Test(resource.Resource):
            allow_patch = True
            id = resource.Body('id')
            attr = resource.Body('attr')
            nested = resource.Body('renamed')
            other = resource.Body('other')
        test_patch = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/nested/dog', 'op': 'remove'}, {'path': '/nested/cat', 'op': 'add', 'value': 'meow'}]
        expected = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/renamed/dog', 'op': 'remove'}, {'path': '/renamed/cat', 'op': 'add', 'value': 'meow'}]
        sot = Test.existing(id=1, attr=42, nested={'dog': 'bark'})
        sot.patch(self.session, test_patch)
        self.session.patch.assert_called_once_with('/1', json=expected, headers=mock.ANY, microversion=None)

    def test_patch_with_server_names(self):

        class Test(resource.Resource):
            allow_patch = True
            id = resource.Body('id')
            attr = resource.Body('attr')
            nested = resource.Body('renamed')
            other = resource.Body('other')
        test_patch = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/renamed/dog', 'op': 'remove'}, {'path': '/renamed/cat', 'op': 'add', 'value': 'meow'}]
        sot = Test.existing(id=1, attr=42, nested={'dog': 'bark'})
        sot.patch(self.session, test_patch)
        self.session.patch.assert_called_once_with('/1', json=test_patch, headers=mock.ANY, microversion=None)

    def test_patch_with_changed_fields(self):

        class Test(resource.Resource):
            allow_patch = True
            attr = resource.Body('attr')
            nested = resource.Body('renamed')
            other = resource.Body('other')
        sot = Test.existing(id=1, attr=42, nested={'dog': 'bark'})
        sot.attr = 'new'
        sot.patch(self.session, {'path': '/renamed/dog', 'op': 'remove'})
        expected = [{'path': '/attr', 'op': 'replace', 'value': 'new'}, {'path': '/renamed/dog', 'op': 'remove'}]
        self.session.patch.assert_called_once_with('/1', json=expected, headers=mock.ANY, microversion=None)

    def test_delete(self):
        result = self.sot.delete(self.session)
        self.sot._prepare_request.assert_called_once_with()
        self.session.delete.assert_called_once_with(self.request.url, headers='headers', microversion=None)
        self.sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, self.sot)

    def test_delete_with_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_delete = True
            _max_microversion = '1.42'
        sot = Test(id='id')
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        result = sot.delete(self.session)
        sot._prepare_request.assert_called_once_with()
        self.session.delete.assert_called_once_with(self.request.url, headers='headers', microversion='1.42')
        sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, sot)

    def test_delete_with_explicit_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            allow_delete = True
            _max_microversion = '1.99'
        sot = Test(id='id')
        sot._prepare_request = mock.Mock(return_value=self.request)
        sot._translate_response = mock.Mock()
        result = sot.delete(self.session, microversion='1.42')
        sot._prepare_request.assert_called_once_with()
        self.session.delete.assert_called_once_with(self.request.url, headers='headers', microversion='1.42')
        sot._translate_response.assert_called_once_with(self.response, has_body=False)
        self.assertEqual(result, sot)

    def test_list_empty_response(self):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'resources': []}
        self.session.get.return_value = mock_response
        result = list(self.sot.list(self.session))
        self.session.get.assert_called_once_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        self.assertEqual([], result)

    def test_list_one_page_response_paginated(self):
        id_value = 1
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': id_value}]}
        self.session.get.return_value = mock_response
        results = list(self.sot.list(self.session, paginated=True))
        self.assertEqual(1, len(results))
        self.assertEqual(1, len(self.session.get.call_args_list))
        self.assertEqual(id_value, results[0].id)
        self.assertIsInstance(results[0], self.test_class)

    def test_list_one_page_response_not_paginated(self):
        id_value = 1
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'resources': [{'id': id_value}]}
        self.session.get.return_value = mock_response
        results = list(self.sot.list(self.session, paginated=False))
        self.session.get.assert_called_once_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        self.assertEqual(1, len(results))
        self.assertEqual(id_value, results[0].id)
        self.assertIsInstance(results[0], self.test_class)

    def test_list_one_page_response_resources_key(self):
        key = 'resources'

        class Test(self.test_class):
            resources_key = key
        id_value = 1
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {key: [{'id': id_value}]}
        mock_response.links = []
        self.session.get.return_value = mock_response
        sot = Test()
        results = list(sot.list(self.session))
        self.session.get.assert_called_once_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        self.assertEqual(1, len(results))
        self.assertEqual(id_value, results[0].id)
        self.assertIsInstance(results[0], self.test_class)

    def test_list_response_paginated_without_links(self):
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
        mock_response2 = mock.Mock()
        mock_response2.status_code = 200
        mock_response2.links = {}
        mock_response2.json.return_value = {'resources': [{'id': ids[1]}]}
        self.session.get.side_effect = [mock_response, mock_response2]
        results = list(self.sot.list(self.session, paginated=True))
        self.assertEqual(2, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertEqual(ids[1], results[1].id)
        self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={}, microversion=None), self.session.get.mock_calls[0])
        self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None), self.session.get.mock_calls[1])
        self.assertEqual(2, len(self.session.get.call_args_list))
        self.assertIsInstance(results[0], self.test_class)

    def test_list_response_paginated_with_links(self):
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.side_effect = [{'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}, {'resources': [{'id': ids[1]}]}]
        self.session.get.return_value = mock_response
        results = list(self.sot.list(self.session, paginated=True))
        self.assertEqual(2, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertEqual(ids[1], results[1].id)
        self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={}, microversion=None), self.session.get.mock_calls[0])
        self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None), self.session.get.mock_calls[2])
        self.assertEqual(2, len(self.session.get.call_args_list))
        self.assertIsInstance(results[0], self.test_class)

    def test_list_response_paginated_with_links_and_query(self):
        q_limit = 1
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.side_effect = [{'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url?limit=%d' % q_limit, 'rel': 'next'}]}, {'resources': [{'id': ids[1]}]}, {'resources': []}]
        self.session.get.return_value = mock_response

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters('limit')
        results = list(Test.list(self.session, paginated=True, limit=q_limit))
        self.assertEqual(2, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertEqual(ids[1], results[1].id)
        self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={'limit': q_limit}, microversion=None), self.session.get.mock_calls[0])
        self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={'limit': [str(q_limit)]}, microversion=None), self.session.get.mock_calls[2])
        self.assertEqual(3, len(self.session.get.call_args_list))
        self.assertIsInstance(results[0], self.test_class)

    def test_list_response_paginated_with_next_field(self):
        """Test pagination with a 'next' field in the response.

        Glance doesn't return a 'links' field in the response. Instead, it
        returns a 'first' field and, if there are more pages, a 'next' field in
        the response body. Ensure we correctly parse these.
        """

        class Test(resource.Resource):
            service = self.service_name
            base_path = '/foos/bars'
            resources_key = 'bars'
            allow_list = True
            _query_mapping = resource.QueryParameters('wow')
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.side_effect = [{'bars': [{'id': ids[0]}], 'first': '/v2/foos/bars?wow=cool', 'next': '/v2/foos/bars?marker=baz&wow=cool'}, {'bars': [{'id': ids[1]}], 'first': '/v2/foos/bars?wow=cool'}]
        self.session.get.return_value = mock_response
        results = list(Test.list(self.session, paginated=True, wow='cool'))
        self.assertEqual(2, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertEqual(ids[1], results[1].id)
        self.assertEqual(mock.call(Test.base_path, headers={'Accept': 'application/json'}, params={'wow': 'cool'}, microversion=None), self.session.get.mock_calls[0])
        self.assertEqual(mock.call('/foos/bars', headers={'Accept': 'application/json'}, params={'wow': ['cool'], 'marker': ['baz']}, microversion=None), self.session.get.mock_calls[2])
        self.assertEqual(2, len(self.session.get.call_args_list))
        self.assertIsInstance(results[0], Test)

    def test_list_response_paginated_with_microversions(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            resources_key = 'resources'
            allow_list = True
            _max_microversion = '1.42'
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
        mock_response2 = mock.Mock()
        mock_response2.status_code = 200
        mock_response2.links = {}
        mock_response2.json.return_value = {'resources': [{'id': ids[1]}]}
        self.session.get.side_effect = [mock_response, mock_response2]
        results = list(Test.list(self.session, paginated=True))
        self.assertEqual(2, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertEqual(ids[1], results[1].id)
        self.assertEqual(mock.call('base_path', headers={'Accept': 'application/json'}, params={}, microversion='1.42'), self.session.get.mock_calls[0])
        self.assertEqual(mock.call('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion='1.42'), self.session.get.mock_calls[1])
        self.assertEqual(2, len(self.session.get.call_args_list))
        self.assertIsInstance(results[0], Test)
        self.assertEqual('1.42', results[0].microversion)

    def test_list_multi_page_response_not_paginated(self):
        ids = [1, 2]
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [{'resources': [{'id': ids[0]}]}, {'resources': [{'id': ids[1]}]}]
        self.session.get.return_value = mock_response
        results = list(self.sot.list(self.session, paginated=False))
        self.assertEqual(1, len(results))
        self.assertEqual(ids[0], results[0].id)
        self.assertIsInstance(results[0], self.test_class)

    def test_list_paginated_infinite_loop(self):
        q_limit = 1
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.side_effect = [{'resources': [{'id': 1}]}, {'resources': [{'id': 1}]}]
        self.session.get.return_value = mock_response

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters('limit')
        res = Test.list(self.session, paginated=True, limit=q_limit)
        self.assertRaises(exceptions.SDKException, list, res)

    def test_list_query_params(self):
        id = 1
        qp = 'query param!'
        qp_name = 'query-param'
        uri_param = 'uri param!'
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': id}]}
        mock_empty = mock.Mock()
        mock_empty.status_code = 200
        mock_empty.links = {}
        mock_empty.json.return_value = {'resources': []}
        self.session.get.side_effect = [mock_response, mock_empty]

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters(query_param=qp_name)
            base_path = '/%(something)s/blah'
            something = resource.URI('something')
        results = list(Test.list(self.session, paginated=True, query_param=qp, something=uri_param))
        self.assertEqual(1, len(results))
        self.assertEqual(results[0].something, uri_param)
        self.assertEqual(self.session.get.call_args_list[0][1]['params'], {qp_name: qp})
        self.assertEqual(self.session.get.call_args_list[0][0][0], Test.base_path % {'something': uri_param})

    def test_allow_invalid_list_params(self):
        qp = 'query param!'
        qp_name = 'query-param'
        uri_param = 'uri param!'
        mock_empty = mock.Mock()
        mock_empty.status_code = 200
        mock_empty.links = {}
        mock_empty.json.return_value = {'resources': []}
        self.session.get.side_effect = [mock_empty]

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters(query_param=qp_name)
            base_path = '/%(something)s/blah'
            something = resource.URI('something')
        list(Test.list(self.session, paginated=True, query_param=qp, allow_unknown_params=True, something=uri_param, something_wrong=True))
        self.session.get.assert_called_once_with('/{something}/blah'.format(something=uri_param), headers={'Accept': 'application/json'}, microversion=None, params={qp_name: qp})

    def test_list_client_filters(self):
        qp = 'query param!'
        uri_param = 'uri param!'
        mock_empty = mock.Mock()
        mock_empty.status_code = 200
        mock_empty.links = {}
        mock_empty.json.return_value = {'resources': [{'a': '1', 'b': '1'}, {'a': '1', 'b': '2'}]}
        self.session.get.side_effect = [mock_empty]

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters('a')
            base_path = '/%(something)s/blah'
            something = resource.URI('something')
            a = resource.Body('a')
            b = resource.Body('b')
        res = list(Test.list(self.session, paginated=True, query_param=qp, allow_unknown_params=True, something=uri_param, a='1', b='2'))
        self.session.get.assert_called_once_with('/{something}/blah'.format(something=uri_param), headers={'Accept': 'application/json'}, microversion=None, params={'a': '1'})
        self.assertEqual(1, len(res))
        self.assertEqual('2', res[0].b)

    def test_values_as_list_params(self):
        id = 1
        qp = 'query param!'
        qp_name = 'query-param'
        uri_param = 'uri param!'
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': id}]}
        mock_empty = mock.Mock()
        mock_empty.status_code = 200
        mock_empty.links = {}
        mock_empty.json.return_value = {'resources': []}
        self.session.get.side_effect = [mock_response, mock_empty]

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters(query_param=qp_name)
            base_path = '/%(something)s/blah'
            something = resource.URI('something')
        results = list(Test.list(self.session, paginated=True, something=uri_param, **{qp_name: qp}))
        self.assertEqual(1, len(results))
        self.assertEqual(self.session.get.call_args_list[0][1]['params'], {qp_name: qp})
        self.assertEqual(self.session.get.call_args_list[0][0][0], Test.base_path % {'something': uri_param})

    def test_values_as_list_params_precedence(self):
        id = 1
        qp = 'query param!'
        qp2 = 'query param!!!!!'
        qp_name = 'query-param'
        uri_param = 'uri param!'
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = {'resources': [{'id': id}]}
        mock_empty = mock.Mock()
        mock_empty.status_code = 200
        mock_empty.links = {}
        mock_empty.json.return_value = {'resources': []}
        self.session.get.side_effect = [mock_response, mock_empty]

        class Test(self.test_class):
            _query_mapping = resource.QueryParameters(query_param=qp_name)
            base_path = '/%(something)s/blah'
            something = resource.URI('something')
        results = list(Test.list(self.session, paginated=True, query_param=qp2, something=uri_param, **{qp_name: qp}))
        self.assertEqual(1, len(results))
        self.assertEqual(self.session.get.call_args_list[0][1]['params'], {qp_name: qp2})
        self.assertEqual(self.session.get.call_args_list[0][0][0], Test.base_path % {'something': uri_param})

    def test_list_multi_page_response_paginated(self):
        ids = [1, 2]
        resp1 = mock.Mock()
        resp1.status_code = 200
        resp1.links = {}
        resp1.json.return_value = {'resources': [{'id': ids[0]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
        resp2 = mock.Mock()
        resp2.status_code = 200
        resp2.links = {}
        resp2.json.return_value = {'resources': [{'id': ids[1]}], 'resources_links': [{'href': 'https://example.com/next-url', 'rel': 'next'}]}
        resp3 = mock.Mock()
        resp3.status_code = 200
        resp3.links = {}
        resp3.json.return_value = {'resources': []}
        self.session.get.side_effect = [resp1, resp2, resp3]
        results = self.sot.list(self.session, paginated=True)
        result0 = next(results)
        self.assertEqual(result0.id, ids[0])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        result1 = next(results)
        self.assertEqual(result1.id, ids[1])
        self.session.get.assert_called_with('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None)
        self.assertRaises(StopIteration, next, results)
        self.session.get.assert_called_with('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None)

    def test_list_multi_page_no_early_termination(self):
        ids = [1, 2, 3, 4]
        resp1 = mock.Mock()
        resp1.status_code = 200
        resp1.links = {}
        resp1.json.return_value = {'resources': [{'id': ids[0]}, {'id': ids[1]}]}
        resp2 = mock.Mock()
        resp2.status_code = 200
        resp2.links = {}
        resp2.json.return_value = {'resources': [{'id': ids[2]}, {'id': ids[3]}]}
        resp3 = mock.Mock()
        resp3.status_code = 200
        resp3.json.return_value = {'resources': []}
        self.session.get.side_effect = [resp1, resp2, resp3]
        results = self.sot.list(self.session, limit=3, paginated=True)
        result0 = next(results)
        self.assertEqual(result0.id, ids[0])
        result1 = next(results)
        self.assertEqual(result1.id, ids[1])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 3}, microversion=None)
        result2 = next(results)
        self.assertEqual(result2.id, ids[2])
        result3 = next(results)
        self.assertEqual(result3.id, ids[3])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 3, 'marker': 2}, microversion=None)
        self.assertRaises(StopIteration, next, results)
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 3, 'marker': 4}, microversion=None)
        self.assertEqual(3, len(self.session.get.call_args_list))

    def test_list_multi_page_inferred_additional(self):
        ids = [1, 2, 3]
        resp1 = mock.Mock()
        resp1.status_code = 200
        resp1.links = {}
        resp1.json.return_value = {'resources': [{'id': ids[0]}, {'id': ids[1]}]}
        resp2 = mock.Mock()
        resp2.status_code = 200
        resp2.links = {}
        resp2.json.return_value = {'resources': [{'id': ids[2]}]}
        self.session.get.side_effect = [resp1, resp2]
        results = self.sot.list(self.session, limit=2, paginated=True)
        result0 = next(results)
        self.assertEqual(result0.id, ids[0])
        result1 = next(results)
        self.assertEqual(result1.id, ids[1])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 2}, microversion=None)
        result2 = next(results)
        self.assertEqual(result2.id, ids[2])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'limit': 2, 'marker': 2}, microversion=None)
        self.assertRaises((StopIteration, RuntimeError), next, results)
        self.assertEqual(3, len(self.session.get.call_args_list))

    def test_list_multi_page_header_count(self):

        class Test(self.test_class):
            resources_key = None
            pagination_key = 'X-Container-Object-Count'
        self.sot = Test()
        ids = [1, 2, 3]
        resp1 = mock.Mock()
        resp1.status_code = 200
        resp1.links = {}
        resp1.headers = {'X-Container-Object-Count': 3}
        resp1.json.return_value = [{'id': ids[0]}, {'id': ids[1]}]
        resp2 = mock.Mock()
        resp2.status_code = 200
        resp2.links = {}
        resp2.headers = {'X-Container-Object-Count': 3}
        resp2.json.return_value = [{'id': ids[2]}]
        self.session.get.side_effect = [resp1, resp2]
        results = self.sot.list(self.session, paginated=True)
        result0 = next(results)
        self.assertEqual(result0.id, ids[0])
        result1 = next(results)
        self.assertEqual(result1.id, ids[1])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        result2 = next(results)
        self.assertEqual(result2.id, ids[2])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={'marker': 2}, microversion=None)
        self.assertRaises(StopIteration, next, results)
        self.assertEqual(2, len(self.session.get.call_args_list))

    def test_list_multi_page_link_header(self):
        ids = [1, 2, 3]
        resp1 = mock.Mock()
        resp1.status_code = 200
        resp1.links = {'next': {'uri': 'https://example.com/next-url', 'rel': 'next'}}
        resp1.headers = {}
        resp1.json.return_value = {'resources': [{'id': ids[0]}, {'id': ids[1]}]}
        resp2 = mock.Mock()
        resp2.status_code = 200
        resp2.links = {}
        resp2.headers = {}
        resp2.json.return_value = {'resources': [{'id': ids[2]}]}
        self.session.get.side_effect = [resp1, resp2]
        results = self.sot.list(self.session, paginated=True)
        result0 = next(results)
        self.assertEqual(result0.id, ids[0])
        result1 = next(results)
        self.assertEqual(result1.id, ids[1])
        self.session.get.assert_called_with(self.base_path, headers={'Accept': 'application/json'}, params={}, microversion=None)
        result2 = next(results)
        self.assertEqual(result2.id, ids[2])
        self.session.get.assert_called_with('https://example.com/next-url', headers={'Accept': 'application/json'}, params={}, microversion=None)
        self.assertRaises(StopIteration, next, results)
        self.assertEqual(2, len(self.session.get.call_args_list))

    def test_bulk_create_invalid_data_passed(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
        Test._prepare_request = mock.Mock()
        self.assertRaises(ValueError, Test.bulk_create, self.session, [])
        self.assertRaises(ValueError, Test.bulk_create, self.session, None)
        self.assertRaises(ValueError, Test.bulk_create, self.session, object)
        self.assertRaises(ValueError, Test.bulk_create, self.session, {})
        self.assertRaises(ValueError, Test.bulk_create, self.session, 'hi!')
        self.assertRaises(ValueError, Test.bulk_create, self.session, ['hi!'])

    def _test_bulk_create(self, cls, http_method, microversion=None, base_path=None, **params):
        req1 = mock.Mock()
        req2 = mock.Mock()
        req1.body = {'name': 'resource1'}
        req2.body = {'name': 'resource2'}
        req1.url = 'uri'
        req2.url = 'uri'
        req1.headers = 'headers'
        req2.headers = 'headers'
        request_body = {'tests': [{'name': 'resource1', 'id': 'id1'}, {'name': 'resource2', 'id': 'id2'}]}
        cls._prepare_request = mock.Mock(side_effect=[req1, req2])
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.links = {}
        mock_response.json.return_value = request_body
        http_method.return_value = mock_response
        res = list(cls.bulk_create(self.session, [{'name': 'resource1'}, {'name': 'resource2'}], base_path=base_path, **params))
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].id, 'id1')
        self.assertEqual(res[1].id, 'id2')
        http_method.assert_called_once_with(self.request.url, json={'tests': [req1.body, req2.body]}, headers=self.request.headers, microversion=microversion, params=params)

    def test_bulk_create_post(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
            resources_key = 'tests'
        self._test_bulk_create(Test, self.session.post)

    def test_bulk_create_put(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'PUT'
            allow_create = True
            resources_key = 'tests'
        self._test_bulk_create(Test, self.session.put)

    def test_bulk_create_with_params(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
            resources_key = 'tests'
        self._test_bulk_create(Test, self.session.post, answer=42)

    def test_bulk_create_with_microversion(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
            resources_key = 'tests'
            _max_microversion = '1.42'
        self._test_bulk_create(Test, self.session.post, microversion='1.42')

    def test_bulk_create_with_base_path(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
            resources_key = 'tests'
        self._test_bulk_create(Test, self.session.post, base_path='dummy')

    def test_bulk_create_fail(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = False
            resources_key = 'tests'
        self.assertRaises(exceptions.MethodNotSupported, Test.bulk_create, self.session, [{'name': 'name'}])

    def test_bulk_create_fail_on_request(self):

        class Test(resource.Resource):
            service = self.service_name
            base_path = self.base_path
            create_method = 'POST'
            allow_create = True
            resources_key = 'tests'
        response = FakeResponse({}, status_code=409)
        response.content = '{"TestError": {"message": "Failed to parse request. Required attribute \'foo\' not specified", "type": "HTTPBadRequest", "detail": ""}}'
        response.reason = 'Bad Request'
        self.session.post.return_value = response
        self.assertRaises(exceptions.ConflictException, Test.bulk_create, self.session, [{'name': 'name'}])
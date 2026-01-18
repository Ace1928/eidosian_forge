import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
class GoogleStorageTests(S3Tests, GoogleTestCase):
    driver_type = google_storage.GoogleStorageDriver
    driver_args = STORAGE_GOOGLE_STORAGE_PARAMS
    mock_response_klass = GoogleStorageMockHttp

    def setUp(self):
        super().setUp()
        self.driver_type.jsonConnectionCls.conn_class = GoogleStorageJSONMockHttp

    def tearDown(self):
        self._remove_test_file()

    def test_billing_not_enabled(self):
        pass

    def test_token(self):
        pass

    def test_get_object_object_size_in_content_length(self):
        self.mock_response_klass.type = 'get_object'
        obj = self.driver.get_object(container_name='test2', object_name='test')
        self.assertEqual(obj.size, 12345)

    def test_get_object_object_size_not_in_content_length_header(self):
        self.mock_response_klass.type = 'get_object'
        obj = self.driver.get_object(container_name='test2', object_name='test_cont_length')
        self.assertEqual(obj.size, 9587)

    def test_delete_permissions(self):
        mock_request = mock.Mock()
        self.driver.json_connection.request = mock_request
        self.driver.ex_delete_permissions('bucket', 'object', entity='user-foo')
        url = '/storage/v1/b/bucket/o/object/acl/user-foo'
        mock_request.assert_called_once_with(url, method='DELETE')
        mock_request.reset_mock()
        self.driver.ex_delete_permissions('bucket', entity='user-foo')
        url = '/storage/v1/b/bucket/acl/user-foo'
        mock_request.assert_called_once_with(url, method='DELETE')

    def test_delete_permissions_no_entity(self):
        mock_request = mock.Mock()
        mock_get_user = mock.Mock(return_value=None)
        self.driver._get_user = mock_get_user
        self.driver.json_connection.request = mock_request
        self.assertRaises(ValueError, self.driver.ex_delete_permissions, 'bucket', 'object')
        self.assertRaises(ValueError, self.driver.ex_delete_permissions, 'bucket')
        mock_request.assert_not_called()
        mock_get_user.return_value = 'foo@foo.com'
        self.driver.ex_delete_permissions('bucket', 'object')
        url = '/storage/v1/b/bucket/o/object/acl/user-foo@foo.com'
        mock_request.assert_called_once_with(url, method='DELETE')
        mock_request.reset_mock()
        mock_get_user.return_value = 'foo@foo.com'
        self.driver.ex_delete_permissions('bucket')
        url = '/storage/v1/b/bucket/acl/user-foo@foo.com'
        mock_request.assert_called_once_with(url, method='DELETE')

    def test_get_permissions(self):

        def test_permission_config(bucket_perms, object_perms):
            GoogleStorageJSONMockHttp.bucket_perms = bucket_perms
            GoogleStorageJSONMockHttp.object_perms = object_perms
            perms = self.driver.ex_get_permissions('test-bucket', 'test-object')
            self.assertEqual(perms, (bucket_perms, object_perms))
        bucket_levels = range(len(google_storage.ContainerPermissions.values))
        object_levels = range(len(google_storage.ObjectPermissions.values))
        for bucket_perms in bucket_levels:
            for object_perms in object_levels:
                test_permission_config(bucket_perms, object_perms)

    def test_set_permissions(self):
        mock_request = mock.Mock()
        self.driver.json_connection.request = mock_request
        self.driver.ex_set_permissions('bucket', 'object', entity='user-foo', role='OWNER')
        url = '/storage/v1/b/bucket/o/object/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
        mock_request.reset_mock()
        self.driver.ex_set_permissions('bucket', 'object', entity='user-foo', role=google_storage.ObjectPermissions.OWNER)
        url = '/storage/v1/b/bucket/o/object/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
        mock_request.reset_mock()
        self.driver.ex_set_permissions('bucket', entity='user-foo', role='OWNER')
        url = '/storage/v1/b/bucket/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
        mock_request.reset_mock()
        self.driver.ex_set_permissions('bucket', entity='user-foo', role=google_storage.ContainerPermissions.OWNER)
        url = '/storage/v1/b/bucket/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))

    def test_set_permissions_bad_roles(self):
        mock_request = mock.Mock()
        self.driver.json_connection.request = mock_request
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object')
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket')
        mock_request.assert_not_called()
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object', role=google_storage.ContainerPermissions.OWNER)
        mock_request.assert_not_called()
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', role=google_storage.ObjectPermissions.OWNER)
        mock_request.assert_not_called()

    def test_set_permissions_no_entity(self):
        mock_request = mock.Mock()
        mock_get_user = mock.Mock(return_value=None)
        self.driver._get_user = mock_get_user
        self.driver.json_connection.request = mock_request
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object', role='OWNER')
        self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', role='OWNER')
        mock_request.assert_not_called()
        mock_get_user.return_value = 'foo@foo.com'
        self.driver.ex_set_permissions('bucket', 'object', role='OWNER')
        url = '/storage/v1/b/bucket/o/object/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo@foo.com'}))
        mock_request.reset_mock()
        mock_get_user.return_value = 'foo@foo.com'
        self.driver.ex_set_permissions('bucket', role='OWNER')
        url = '/storage/v1/b/bucket/acl'
        mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo@foo.com'}))

    def test_invalid_credentials_on_upload(self):
        self.mock_response_klass.type = 'UNAUTHORIZED'
        container = Container(name='container', driver=self.driver, extra={})
        with pytest.raises(InvalidCredsError):
            self.driver.upload_object_via_stream(BytesIO(b' '), container, 'path')

    def test_download_object_data_is_not_buffered_in_memory(self):
        mock_response = Mock(name='mock response')
        mock_response.headers = {}
        mock_response.status_code = 200
        msg = '"content" attribute was accessed but it shouldn\'t have been'
        type(mock_response).content = PropertyMock(name='mock content attribute', side_effect=Exception(msg))
        mock_response.iter_content.return_value = StringIO('a' * 1000)
        self.driver.connection.connection.getresponse = Mock()
        self.driver.connection.connection.getresponse.return_value = mock_response
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object_NO_BUFFER', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = self._file_path
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=True, delete_on_failure=True)
        self.assertTrue(result)
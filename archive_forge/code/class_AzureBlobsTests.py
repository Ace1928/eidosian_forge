import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
class AzureBlobsTests(unittest.TestCase):
    driver_type = AzureBlobsStorageDriver
    driver_args = STORAGE_AZURE_BLOBS_PARAMS
    mock_response_klass = AzureBlobsMockHttp

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args)

    def setUp(self):
        self.driver_type.connectionCls.conn_class = self.mock_response_klass
        self.mock_response_klass.type = None
        self.driver = self.create_driver()

    def tearDown(self):
        self._remove_test_file()

    def _remove_test_file(self):
        file_path = os.path.abspath(__file__) + '.temp'
        try:
            os.unlink(file_path)
        except OSError:
            pass

    def test_invalid_credentials(self):
        self.mock_response_klass.type = 'UNAUTHORIZED'
        try:
            self.driver.list_containers()
        except InvalidCredsError as e:
            self.assertEqual(True, isinstance(e, InvalidCredsError))
        else:
            self.fail('Exception was not thrown')

    def test_list_containers_empty(self):
        self.mock_response_klass.type = 'list_containers_EMPTY'
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 0)

    def test_list_containers_success(self):
        self.mock_response_klass.type = 'list_containers'
        AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 4)
        self.assertTrue('last_modified' in containers[1].extra)
        self.assertTrue('url' in containers[1].extra)
        self.assertTrue('etag' in containers[1].extra)
        self.assertTrue('lease' in containers[1].extra)
        self.assertTrue('meta_data' in containers[1].extra)
        self.assertEqual(containers[1].extra['etag'], '0x8CFBAB7B5B82D8E')

    def test_list_container_objects_empty(self):
        self.mock_response_klass.type = 'EMPTY'
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 0)

    def test_list_container_objects_success(self):
        self.mock_response_klass.type = None
        AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 4)
        obj = objects[1]
        self.assertEqual(obj.name, 'object2.txt')
        self.assertEqual(obj.hash, '0x8CFB90F1BA8CD8F')
        self.assertEqual(obj.size, 1048576)
        self.assertEqual(obj.container.name, 'test_container')
        self.assertTrue('meta1' in obj.meta_data)
        self.assertTrue('meta2' in obj.meta_data)
        self.assertTrue('last_modified' in obj.extra)
        self.assertTrue('content_type' in obj.extra)
        self.assertTrue('content_encoding' in obj.extra)
        self.assertTrue('content_language' in obj.extra)

    def test_list_container_objects_with_prefix(self):
        self.mock_response_klass.type = None
        AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container, prefix='test_prefix')
        self.assertEqual(len(objects), 4)
        obj = objects[1]
        self.assertEqual(obj.name, 'object2.txt')
        self.assertEqual(obj.hash, '0x8CFB90F1BA8CD8F')
        self.assertEqual(obj.size, 1048576)
        self.assertEqual(obj.container.name, 'test_container')
        self.assertTrue('meta1' in obj.meta_data)
        self.assertTrue('meta2' in obj.meta_data)
        self.assertTrue('last_modified' in obj.extra)
        self.assertTrue('content_type' in obj.extra)
        self.assertTrue('content_encoding' in obj.extra)
        self.assertTrue('content_language' in obj.extra)

    def test_get_container_doesnt_exist(self):
        self.mock_response_klass.type = None
        try:
            self.driver.get_container(container_name='test_container100')
        except ContainerDoesNotExistError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_get_container_success(self):
        self.mock_response_klass.type = None
        container = self.driver.get_container(container_name='test_container200')
        self.assertTrue(container.name, 'test_container200')
        self.assertTrue(container.extra['etag'], '0x8CFB877BB56A6FB')
        self.assertTrue(container.extra['last_modified'], 'Fri, 04 Jan 2013 09:48:06 GMT')
        self.assertTrue(container.extra['lease']['status'], 'unlocked')
        self.assertTrue(container.extra['lease']['state'], 'available')
        self.assertTrue(container.extra['meta_data']['meta1'], 'value1')
        if self.driver.secure:
            expected_url = 'https://account.blob.core.windows.net/test_container200'
        else:
            expected_url = 'http://localhost/account/test_container200'
        self.assertEqual(container.extra['url'], expected_url)

    def test_get_object_cdn_url(self):
        obj = self.driver.get_object(container_name='test_container200', object_name='test')
        url = urlparse.urlparse(self.driver.get_object_cdn_url(obj))
        query = urlparse.parse_qs(url.query)
        self.assertEqual(len(query['sig']), 1)
        self.assertGreater(len(query['sig'][0]), 0)

    def test_get_object_container_doesnt_exist(self):
        self.mock_response_klass.type = None
        try:
            self.driver.get_object(container_name='test_container100', object_name='test')
        except ContainerDoesNotExistError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_get_object_success(self):
        self.mock_response_klass.type = None
        obj = self.driver.get_object(container_name='test_container200', object_name='test')
        self.assertEqual(obj.name, 'test')
        self.assertEqual(obj.container.name, 'test_container200')
        self.assertEqual(obj.size, 12345)
        self.assertEqual(obj.hash, '0x8CFB877BB56A6FB')
        self.assertEqual(obj.extra['last_modified'], 'Fri, 04 Jan 2013 09:48:06 GMT')
        self.assertEqual(obj.extra['content_type'], 'application/zip')
        self.assertEqual(obj.meta_data['rabbits'], 'monkeys')

    def test_create_container_invalid_name(self):
        self.mock_response_klass.type = 'INVALID_NAME'
        try:
            self.driver.create_container(container_name='new--container')
        except InvalidContainerNameError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_create_container_already_exists(self):
        self.mock_response_klass.type = 'ALREADY_EXISTS'
        try:
            self.driver.create_container(container_name='new-container')
        except ContainerAlreadyExistsError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_create_container_success(self):
        self.mock_response_klass.type = None
        name = 'new-container'
        container = self.driver.create_container(container_name=name)
        self.assertEqual(container.name, name)

    def test_delete_container_doesnt_exist(self):
        container = Container(name='new_container', extra=None, driver=self.driver)
        self.mock_response_klass.type = 'DOESNT_EXIST'
        try:
            self.driver.delete_container(container=container)
        except ContainerDoesNotExistError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_delete_container_not_empty(self):
        self.mock_response_klass.type = None
        AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
        container = Container(name='test_container', extra={}, driver=self.driver)
        try:
            self.driver.delete_container(container=container)
        except ContainerIsNotEmptyError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_delete_container_success(self):
        self.mock_response_klass.type = 'EMPTY'
        AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
        container = Container(name='test_container', extra={}, driver=self.driver)
        self.assertTrue(self.driver.delete_container(container=container))

    def test_delete_container_not_found(self):
        self.mock_response_klass.type = 'NOT_FOUND'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        try:
            self.driver.delete_container(container=container)
        except ContainerDoesNotExistError:
            pass
        else:
            self.fail('Container does not exist but an exception was not' + 'thrown')

    def test_download_object_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        self.assertTrue(result)

    def test_download_object_invalid_file_size(self):
        self.mock_response_klass.type = 'INVALID_SIZE'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        self.assertFalse(result)

    def test_download_object_invalid_file_already_exists(self):
        self.mock_response_klass.type = 'INVALID_SIZE'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__)
        try:
            self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        except LibcloudError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_download_object_as_stream_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        stream = self.driver.download_object_as_stream(obj=obj, chunk_size=None)
        consumed_stream = ''.join((chunk.decode('utf-8') for chunk in stream))
        self.assertEqual(len(consumed_stream), obj.size)

    def test_download_object_range_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object_range', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        result = self.driver.download_object_range(obj=obj, start_bytes=5, end_bytes=7, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        self.assertTrue(result)
        with open(destination_path) as fp:
            content = fp.read()
        self.assertEqual(content, '56')

    def test_download_object_range_as_stream_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object_range_stream', size=2, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        stream = self.driver.download_object_range_as_stream(obj=obj, start_bytes=4, end_bytes=6, chunk_size=None)
        consumed_stream = ''.join((chunk.decode('utf-8') for chunk in stream))
        self.assertEqual(consumed_stream, '45')
        self.assertEqual(len(consumed_stream), obj.size)

    def test_upload_object_invalid_md5(self):
        self.mock_response_klass.type = 'INVALID_HASH'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        file_path = os.path.abspath(__file__)
        try:
            self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, verify_hash=True)
        except ObjectHashMismatchError:
            pass
        else:
            self.fail('Invalid hash was returned but an exception was not thrown')

    def test_upload_small_block_object_success(self):
        file_path = os.path.abspath(__file__)
        file_size = os.stat(file_path).st_size
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'meta_data': {'some-value': 'foobar'}}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=False)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, file_size)
        self.assertTrue('some-value' in obj.meta_data)

    def test_upload_big_block_object_success(self):
        _, file_path = tempfile.mkstemp(suffix='.jpg')
        file_size = AZURE_UPLOAD_CHUNK_SIZE + 1
        with open(file_path, 'w') as file_hdl:
            file_hdl.write('0' * file_size)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'meta_data': {'some-value': 'foobar'}}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=False)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, file_size)
        self.assertTrue('some-value' in obj.meta_data)
        os.remove(file_path)

    def test_upload_small_block_object_success_with_lease(self):
        self.mock_response_klass.use_param = 'comp'
        file_path = os.path.abspath(__file__)
        file_size = os.stat(file_path).st_size
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'meta_data': {'some-value': 'foobar'}}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=False, ex_use_lease=True)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, file_size)
        self.assertTrue('some-value' in obj.meta_data)
        self.mock_response_klass.use_param = None

    def test_upload_big_block_object_success_with_lease(self):
        self.mock_response_klass.use_param = 'comp'
        _, file_path = tempfile.mkstemp(suffix='.jpg')
        file_size = AZURE_UPLOAD_CHUNK_SIZE * 2
        with open(file_path, 'w') as file_hdl:
            file_hdl.write('0' * file_size)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'meta_data': {'some-value': 'foobar'}}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=False, ex_use_lease=False)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, file_size)
        self.assertTrue('some-value' in obj.meta_data)
        os.remove(file_path)
        self.mock_response_klass.use_param = None

    def test_upload_blob_object_via_stream(self):
        self.mock_response_klass.use_param = 'comp'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        iterator = BytesIO(b('345'))
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, 3)
        self.mock_response_klass.use_param = None

    def test_upload_blob_object_via_stream_from_iterable(self):
        self.mock_response_klass.use_param = 'comp'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        iterator = iter([b('34'), b('5')])
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, 3)
        self.mock_response_klass.use_param = None

    def test_upload_blob_object_via_stream_with_lease(self):
        self.mock_response_klass.use_param = 'comp'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        iterator = BytesIO(b('345'))
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra, ex_use_lease=True)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, 3)
        self.mock_response_klass.use_param = None

    def test_delete_object_not_found(self):
        self.mock_response_klass.type = 'NOT_FOUND'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1234, hash=None, extra=None, meta_data=None, container=container, driver=self.driver)
        try:
            self.driver.delete_object(obj=obj)
        except ObjectDoesNotExistError:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_delete_object_success(self):
        self.mock_response_klass.type = 'DELETE'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1234, hash=None, extra=None, meta_data=None, container=container, driver=self.driver)
        result = self.driver.delete_object(obj=obj)
        self.assertTrue(result)

    def test_storage_driver_host(self):
        driver1 = self.driver_type('fakeaccount1', 'deadbeafcafebabe==')
        driver2 = self.driver_type('fakeaccount2', 'deadbeafcafebabe==')
        driver3 = self.driver_type('fakeaccount3', 'deadbeafcafebabe==', host='test.foo.bar.com')
        host1 = driver1.connection.host
        host2 = driver2.connection.host
        host3 = driver3.connection.host
        self.assertEqual(host1, 'fakeaccount1.blob.core.windows.net')
        self.assertEqual(host2, 'fakeaccount2.blob.core.windows.net')
        self.assertEqual(host3, 'test.foo.bar.com')

    def test_normalize_http_headers(self):
        driver = self.driver_type('fakeaccount1', 'deadbeafcafebabe==')
        headers = driver._fix_headers({'Content-Encoding': 'gzip', 'content-language': 'en-us', 'x-foo': 'bar'})
        self.assertEqual(headers, {'x-ms-blob-content-encoding': 'gzip', 'x-ms-blob-content-language': 'en-us', 'x-foo': 'bar'})

    def test_storage_driver_host_govcloud(self):
        driver1 = self.driver_type('fakeaccount1', 'deadbeafcafebabe==', host='blob.core.usgovcloudapi.net')
        driver2 = self.driver_type('fakeaccount2', 'deadbeafcafebabe==', host='fakeaccount2.blob.core.usgovcloudapi.net')
        host1 = driver1.connection.host
        host2 = driver2.connection.host
        account_prefix_1 = driver1.connection.account_prefix
        account_prefix_2 = driver2.connection.account_prefix
        self.assertEqual(host1, 'fakeaccount1.blob.core.usgovcloudapi.net')
        self.assertEqual(host2, 'fakeaccount2.blob.core.usgovcloudapi.net')
        self.assertIsNone(account_prefix_1)
        self.assertIsNone(account_prefix_2)

    def test_storage_driver_host_azurite(self):
        driver = self.driver_type('fakeaccount1', 'deadbeafcafebabe==', host='localhost', port=10000, secure=False)
        host = driver.connection.host
        account_prefix = driver.connection.account_prefix
        self.assertEqual(host, 'localhost')
        self.assertEqual(account_prefix, 'fakeaccount1')

    def test_storage_driver_azure_ad(self):
        AzureBlobsActiveDirectoryConnection.conn_class = AzureBlobsMockHttp
        driver = self.driver_type(key='fakeaccount1', secret='DEKjfhdakkdjfhei~', tenant_id='77777777-7777-7777-7777-777777777777', identity='55555555-5555-5555-5555-555555555555', auth_type='azureAd', secure=True)
        host = driver.connection.host
        self.assertEqual(host, 'fakeaccount1.blob.core.windows.net')

    def test_get_azure_ad_object_success(self):
        AzureBlobsActiveDirectoryConnection.conn_class = AzureBlobsMockHttp
        driver = self.driver_type(key='fakeaccount1', secret='DEKjfhdakkdjfhei~', tenant_id='77777777-7777-7777-7777-777777777777', identity='55555555-5555-5555-5555-555555555555', auth_type='azureAd', secure=True)
        self.mock_response_klass.type = None
        container = driver.get_container(container_name='test_container200')
        self.assertTrue(container.name, 'test_container200')
        self.assertTrue(container.extra['etag'], '0x8CFB877BB56A6FB')
        self.assertTrue(container.extra['last_modified'], 'Fri, 04 Jan 2013 09:48:06 GMT')
        self.assertTrue(container.extra['lease']['status'], 'unlocked')
        self.assertTrue(container.extra['lease']['state'], 'available')
        self.assertTrue(container.extra['meta_data']['meta1'], 'value1')
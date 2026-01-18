import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
class OSSStorageDriverTestCase(unittest.TestCase):
    driver_type = OSSStorageDriver
    driver_args = STORAGE_OSS_PARAMS
    mock_response_klass = OSSMockHttp

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args)

    def setUp(self):
        self.driver_type.connectionCls.conn_class = self.mock_response_klass
        self.mock_response_klass.type = None
        self.mock_response_klass.test = self
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
        self.mock_response_klass.type = 'unauthorized'
        self.assertRaises(InvalidCredsError, self.driver.list_containers)

    def test_list_containers_empty(self):
        self.mock_response_klass.type = 'list_containers_empty'
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 0)

    def test_list_containers_success(self):
        self.mock_response_klass.type = 'list_containers'
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 2)
        container = containers[0]
        self.assertEqual('xz02tphky6fjfiuc0', container.name)
        self.assertTrue('creation_date' in container.extra)
        self.assertEqual('2014-05-15T11:18:32.000Z', container.extra['creation_date'])
        self.assertTrue('location' in container.extra)
        self.assertEqual('oss-cn-hangzhou-a', container.extra['location'])
        self.assertEqual(self.driver, container.driver)

    def test_list_container_objects_empty(self):
        self.mock_response_klass.type = 'list_container_objects_empty'
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 0)

    def test_list_container_objects_success(self):
        self.mock_response_klass.type = 'list_container_objects'
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 2)
        obj = objects[0]
        self.assertEqual(obj.name, 'en/')
        self.assertEqual(obj.hash, 'D41D8CD98F00B204E9800998ECF8427E')
        self.assertEqual(obj.size, 0)
        self.assertEqual(obj.container.name, 'test_container')
        self.assertEqual(obj.extra['last_modified'], '2016-01-15T14:43:15.000Z')
        self.assertTrue('owner' in obj.meta_data)

    def test_list_container_objects_with_chinese(self):
        self.mock_response_klass.type = 'list_container_objects_chinese'
        container = Container(name='test_container', extra={}, driver=self.driver)
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 2)
        obj = [o for o in objects if o.name == 'WEB控制台.odp'][0]
        self.assertEqual(obj.hash, '281371EA1618CF0E645D6BB90A158276')
        self.assertEqual(obj.size, 1234567)
        self.assertEqual(obj.container.name, 'test_container')
        self.assertEqual(obj.extra['last_modified'], '2016-01-15T14:43:06.000Z')
        self.assertTrue('owner' in obj.meta_data)

    def test_list_container_objects_with_prefix(self):
        self.mock_response_klass.type = 'list_container_objects_prefix'
        container = Container(name='test_container', extra={}, driver=self.driver)
        self.prefix = 'test_prefix'
        objects = self.driver.list_container_objects(container=container, prefix=self.prefix)
        self.assertEqual(len(objects), 2)

    def test_get_container_doesnt_exist(self):
        self.mock_response_klass.type = 'get_container'
        self.assertRaises(ContainerDoesNotExistError, self.driver.get_container, container_name='not-existed')

    def test_get_container_success(self):
        self.mock_response_klass.type = 'get_container'
        container = self.driver.get_container(container_name='xz02tphky6fjfiuc0')
        self.assertTrue(container.name, 'xz02tphky6fjfiuc0')

    def test_get_object_container_doesnt_exist(self):
        self.mock_response_klass.type = 'get_object'
        self.assertRaises(ObjectDoesNotExistError, self.driver.get_object, container_name='xz02tphky6fjfiuc0', object_name='notexisted')

    def test_get_object_success(self):
        self.mock_response_klass.type = 'get_object'
        obj = self.driver.get_object(container_name='xz02tphky6fjfiuc0', object_name='test')
        self.assertEqual(obj.name, 'test')
        self.assertEqual(obj.container.name, 'xz02tphky6fjfiuc0')
        self.assertEqual(obj.size, 0)
        self.assertEqual(obj.hash, 'D41D8CD98F00B204E9800998ECF8427E')
        self.assertEqual(obj.extra['last_modified'], 'Fri, 15 Jan 2016 14:43:15 GMT')
        self.assertEqual(obj.extra['content_type'], 'application/octet-stream')
        self.assertEqual(obj.meta_data['rabbits'], 'monkeys')

    def test_create_container_bad_request(self):
        self.mock_response_klass.type = 'invalid_name'
        self.assertRaises(ContainerError, self.driver.create_container, container_name='invalid_name')

    def test_create_container_already_exists(self):
        self.mock_response_klass.type = 'already_exists'
        self.assertRaises(InvalidContainerNameError, self.driver.create_container, container_name='new-container')

    def test_create_container_success(self):
        self.mock_response_klass.type = 'create_container'
        name = 'new_container'
        container = self.driver.create_container(container_name=name)
        self.assertEqual(container.name, name)

    def test_create_container_with_ex_location(self):
        self.mock_response_klass.type = 'create_container_location'
        name = 'new_container'
        self.ex_location = 'oss-cn-beijing'
        container = self.driver.create_container(container_name=name, ex_location=self.ex_location)
        self.assertEqual(container.name, name)
        self.assertTrue(container.extra['location'], self.ex_location)

    def test_delete_container_doesnt_exist(self):
        container = Container(name='new_container', extra=None, driver=self.driver)
        self.mock_response_klass.type = 'delete_container_doesnt_exist'
        self.assertRaises(ContainerDoesNotExistError, self.driver.delete_container, container=container)

    def test_delete_container_not_empty(self):
        container = Container(name='new_container', extra=None, driver=self.driver)
        self.mock_response_klass.type = 'delete_container_not_empty'
        self.assertRaises(ContainerIsNotEmptyError, self.driver.delete_container, container=container)

    def test_delete_container_success(self):
        self.mock_response_klass.type = 'delete_container'
        container = Container(name='new_container', extra=None, driver=self.driver)
        self.assertTrue(self.driver.delete_container(container=container))

    def test_download_object_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        self.assertTrue(result)

    def test_download_object_invalid_file_size(self):
        self.mock_response_klass.type = 'invalid_size'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
        self.assertFalse(result)

    def test_download_object_not_found(self):
        self.mock_response_klass.type = 'not_found'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        destination_path = os.path.abspath(__file__) + '.temp'
        self.assertRaises(ObjectDoesNotExistError, self.driver.download_object, obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)

    def test_download_object_as_stream_success(self):
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
        stream = self.driver.download_object_as_stream(obj=obj, chunk_size=None)
        self.assertTrue(hasattr(stream, '__iter__'))

    def test_upload_object_invalid_hash1(self):

        def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None, container=None):
            return {'response': make_response(200, headers={'etag': '2345'}), 'bytes_transferred': 1000, 'data_hash': 'hash343hhash89h932439jsaa89'}
        self.mock_response_klass.type = 'INVALID_HASH1'
        old_func = self.driver_type._upload_object
        self.driver_type._upload_object = upload_file
        file_path = os.path.abspath(__file__)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        try:
            self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, verify_hash=True)
        except ObjectHashMismatchError:
            pass
        else:
            self.fail('Invalid hash was returned but an exception was not thrown')
        finally:
            self.driver_type._upload_object = old_func

    def test_upload_object_success(self):

        def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None, container=None):
            return {'response': make_response(200, headers={'etag': '0cc175b9c0f1b6a831c399e269772661'}), 'bytes_transferred': 1000, 'data_hash': '0cc175b9c0f1b6a831c399e269772661'}
        self.mock_response_klass.type = None
        old_func = self.driver_type._upload_object
        self.driver_type._upload_object = upload_file
        file_path = os.path.abspath(__file__)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'meta_data': {'some-value': 'foobar'}}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=True)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, 1000)
        self.assertTrue('some-value' in obj.meta_data)
        self.driver_type._upload_object = old_func

    def test_upload_object_with_acl(self):

        def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None, container=None):
            return {'response': make_response(200, headers={'etag': '0cc175b9c0f1b6a831c399e269772661'}), 'bytes_transferred': 1000, 'data_hash': '0cc175b9c0f1b6a831c399e269772661'}
        self.mock_response_klass.type = None
        old_func = self.driver_type._upload_object
        self.driver_type._upload_object = upload_file
        file_path = os.path.abspath(__file__)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'acl': 'public-read'}
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=True)
        self.assertEqual(obj.name, 'foo_test_upload')
        self.assertEqual(obj.size, 1000)
        self.assertEqual(obj.extra['acl'], 'public-read')
        self.driver_type._upload_object = old_func

    def test_upload_object_with_invalid_acl(self):
        file_path = os.path.abspath(__file__)
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_upload'
        extra = {'acl': 'invalid-acl'}
        self.assertRaises(AttributeError, self.driver.upload_object, file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=True)

    def test_upload_empty_object_via_stream(self):
        if self.driver.supports_multipart_upload:
            self.mock_response_klass.type = 'multipart'
        else:
            self.mock_response_klass.type = None
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_stream_data'
        iterator = DummyIterator(data=[''])
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, 0)

    def test_upload_small_object_via_stream(self):
        if self.driver.supports_multipart_upload:
            self.mock_response_klass.type = 'multipart'
        else:
            self.mock_response_klass.type = None
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_stream_data'
        iterator = DummyIterator(data=['2', '3', '5'])
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, 3)

    def test_upload_big_object_via_stream(self):
        if self.driver.supports_multipart_upload:
            self.mock_response_klass.type = 'multipart'
        else:
            self.mock_response_klass.type = None
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_stream_data'
        iterator = DummyIterator(data=['2' * CHUNK_SIZE, '3' * CHUNK_SIZE, '5'])
        extra = {'content_type': 'text/plain'}
        obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        self.assertEqual(obj.name, object_name)
        self.assertEqual(obj.size, CHUNK_SIZE * 2 + 1)

    def test_upload_object_via_stream_abort(self):
        if not self.driver.supports_multipart_upload:
            return
        self.mock_response_klass.type = 'MULTIPART'

        def _faulty_iterator():
            for i in range(0, 5):
                yield str(i)
            raise RuntimeError('Error in fetching data')
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        object_name = 'foo_test_stream_data'
        iterator = _faulty_iterator()
        extra = {'content_type': 'text/plain'}
        try:
            self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
        except Exception:
            pass
        return

    def test_ex_iterate_multipart_uploads(self):
        if not self.driver.supports_multipart_upload:
            return
        self.mock_response_klass.type = 'list_multipart'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        for upload in self.driver.ex_iterate_multipart_uploads(container, max_uploads=2):
            self.assertTrue(upload.key is not None)
            self.assertTrue(upload.id is not None)
            self.assertTrue(upload.initiated is not None)

    def test_ex_abort_all_multipart_uploads(self):
        if not self.driver.supports_multipart_upload:
            return
        self.mock_response_klass.type = 'list_multipart'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        with mock.patch('libcloud.storage.drivers.oss.OSSStorageDriver._abort_multipart', autospec=True) as mock_abort:
            self.driver.ex_abort_all_multipart_uploads(container)
            self.assertEqual(3, mock_abort.call_count)

    def test_delete_object_not_found(self):
        self.mock_response_klass.type = 'not_found'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1234, hash=None, extra=None, meta_data=None, container=container, driver=self.driver)
        self.assertRaises(ObjectDoesNotExistError, self.driver.delete_object, obj=obj)

    def test_delete_object_success(self):
        self.mock_response_klass.type = 'delete'
        container = Container(name='foo_bar_container', extra={}, driver=self.driver)
        obj = Object(name='foo_bar_object', size=1234, hash=None, extra=None, meta_data=None, container=container, driver=self.driver)
        result = self.driver.delete_object(obj=obj)
        self.assertTrue(result)
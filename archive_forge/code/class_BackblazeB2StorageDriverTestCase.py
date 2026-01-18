import os
import sys
import json
import tempfile
from unittest import mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import b, httplib
from libcloud.utils.files import exhaust_iterator
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.backblaze_b2 import BackblazeB2StorageDriver
class BackblazeB2StorageDriverTestCase(unittest.TestCase):
    driver_klass = BackblazeB2StorageDriver
    driver_args = ('a', 'b')

    def setUp(self):
        self.driver_klass.connectionCls.authCls.conn_class = BackblazeB2MockHttp
        self.driver_klass.connectionCls.conn_class = BackblazeB2MockHttp
        BackblazeB2MockHttp.type = None
        self.driver = self.driver_klass(*self.driver_args)

    def test_list_containers(self):
        containers = self.driver.list_containers()
        self.assertEqual(len(containers), 3)
        self.assertEqual(containers[0].name, 'test00001')
        self.assertEqual(containers[0].extra['id'], '481c37de2e1ab3bf5e150710')
        self.assertEqual(containers[0].extra['bucketType'], 'allPrivate')

    def test_list_container_objects(self):
        container = self.driver.list_containers()[0]
        objects = self.driver.list_container_objects(container=container)
        self.assertEqual(len(objects), 4)
        self.assertEqual(objects[0].name, '2.txt')
        self.assertEqual(objects[0].size, 2)
        self.assertEqual(objects[0].extra['fileId'], 'abcd')
        self.assertEqual(objects[0].extra['uploadTimestamp'], 1450545966000)

    def test_get_container(self):
        container = self.driver.get_container('test00001')
        self.assertEqual(container.name, 'test00001')
        self.assertEqual(container.extra['id'], '481c37de2e1ab3bf5e150710')
        self.assertEqual(container.extra['bucketType'], 'allPrivate')

    def test_get_object(self):
        obj = self.driver.get_object('test00001', '2.txt')
        self.assertEqual(obj.name, '2.txt')
        self.assertEqual(obj.size, 2)
        self.assertEqual(obj.extra['fileId'], 'abcd')
        self.assertEqual(obj.extra['uploadTimestamp'], 1450545966000)

    def test_create_container(self):
        container = self.driver.create_container(container_name='test0005')
        self.assertEqual(container.name, 'test0005')
        self.assertEqual(container.extra['id'], '681c87aebeaa530f5e250710')
        self.assertEqual(container.extra['bucketType'], 'allPrivate')

    def test_delete_container(self):
        container = self.driver.list_containers()[0]
        result = self.driver.delete_container(container=container)
        self.assertTrue(result)

    def test_download_object(self):
        container = self.driver.list_containers()[0]
        obj = self.driver.list_container_objects(container=container)[0]
        _, destination_path = tempfile.mkstemp()
        result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=True)
        self.assertTrue(result)

    def test_download_object_as_stream(self):
        container = self.driver.list_containers()[0]
        obj = self.driver.list_container_objects(container=container)[0]
        stream = self.driver.download_object_as_stream(obj=obj, chunk_size=1024)
        self.assertTrue(hasattr(stream, '__iter__'))
        self.assertEqual(exhaust_iterator(stream), b('ab'))

    def test_upload_object(self):
        file_path = os.path.abspath(__file__)
        container = self.driver.list_containers()[0]
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name='test0007.txt')
        self.assertEqual(obj.name, 'test0007.txt')
        self.assertEqual(obj.size, 24)
        self.assertEqual(obj.extra['fileId'], 'abcde')

    def test_upload_object_via_stream(self):
        container = self.driver.list_containers()[0]
        file_path = os.path.abspath(__file__)
        with open(file_path, 'rb') as fp:
            iterator = iter(fp)
            obj = self.driver.upload_object_via_stream(iterator=iterator, container=container, object_name='test0007.txt')
        self.assertEqual(obj.name, 'test0007.txt')
        self.assertEqual(obj.size, 24)
        self.assertEqual(obj.extra['fileId'], 'abcde')

    def test_upload_object_with_metadata(self):
        file_path = os.path.abspath(__file__)
        container = self.driver.list_containers()[0]
        obj = self.driver.upload_object(file_path=file_path, container=container, object_name='test0007.txt', extra={'meta_data': {'foo': 'bar', 'baz': 1}})
        self.assertEqual(obj.name, 'test0007.txt')
        self.assertEqual(obj.size, 24)
        self.assertEqual(obj.extra['fileId'], 'abcde')

    def test_delete_object(self):
        container = self.driver.list_containers()[0]
        obj = self.driver.list_container_objects(container=container)[0]
        result = self.driver.delete_object(obj=obj)
        self.assertTrue(result)

    def test_ex_hide_object(self):
        container = self.driver.list_containers()[0]
        container_id = container.extra['id']
        obj = self.driver.ex_hide_object(container_id=container_id, object_name='2.txt')
        self.assertEqual(obj.name, '2.txt')

    def test_ex_list_object_versions(self):
        container = self.driver.list_containers()[0]
        container_id = container.extra['id']
        objects = self.driver.ex_list_object_versions(container_id=container_id)
        self.assertEqual(len(objects), 9)

    def test_ex_get_upload_data(self):
        container = self.driver.list_containers()[0]
        container_id = container.extra['id']
        data = self.driver.ex_get_upload_data(container_id=container_id)
        self.assertEqual(data['authorizationToken'], 'nope')
        self.assertEqual(data['bucketId'], '481c37de2e1ab3bf5e150710')
        self.assertEqual(data['uploadUrl'], 'https://podxxx.backblaze.com/b2api/v1/b2_upload_file/abcd/defg')

    def test_ex_get_upload_url(self):
        container = self.driver.list_containers()[0]
        container_id = container.extra['id']
        url = self.driver.ex_get_upload_url(container_id=container_id)
        self.assertEqual(url, 'https://podxxx.backblaze.com/b2api/v1/b2_upload_file/abcd/defg')
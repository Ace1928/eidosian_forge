import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
class UploadsTest(unittest.TestCase):
    _DEFAULT_BUCKET = 'apitools'
    _TESTDATA_PREFIX = 'uploads'

    def setUp(self):
        self.__client = _GetClient()
        self.__files = []
        self.__content = ''
        self.__buffer = None
        self.__upload = None

    def tearDown(self):
        self.__DeleteFiles()

    def __ResetUpload(self, size, auto_transfer=True):
        self.__content = ''.join((random.choice(string.ascii_letters) for _ in range(size)))
        self.__buffer = six.StringIO(self.__content)
        self.__upload = storage.Upload.FromStream(self.__buffer, 'text/plain', auto_transfer=auto_transfer)

    def __DeleteFiles(self):
        for filename in self.__files:
            self.__DeleteFile(filename)

    def __DeleteFile(self, filename):
        object_name = os.path.join(self._TESTDATA_PREFIX, filename)
        req = storage.StorageObjectsDeleteRequest(bucket=self._DEFAULT_BUCKET, object=object_name)
        self.__client.objects.Delete(req)

    def __InsertRequest(self, filename):
        object_name = os.path.join(self._TESTDATA_PREFIX, filename)
        return storage.StorageObjectsInsertRequest(name=object_name, bucket=self._DEFAULT_BUCKET)

    def __GetRequest(self, filename):
        object_name = os.path.join(self._TESTDATA_PREFIX, filename)
        return storage.StorageObjectsGetRequest(object=object_name, bucket=self._DEFAULT_BUCKET)

    def __InsertFile(self, filename, request=None):
        if request is None:
            request = self.__InsertRequest(filename)
        response = self.__client.objects.Insert(request, upload=self.__upload)
        self.assertIsNotNone(response)
        self.__files.append(filename)
        return response

    def testZeroBytes(self):
        filename = 'zero_byte_file'
        self.__ResetUpload(0)
        response = self.__InsertFile(filename)
        self.assertEqual(0, response.size)

    def testSimpleUpload(self):
        filename = 'fifteen_byte_file'
        self.__ResetUpload(15)
        response = self.__InsertFile(filename)
        self.assertEqual(15, response.size)

    def testMultipartUpload(self):
        filename = 'fifteen_byte_file'
        self.__ResetUpload(15)
        request = self.__InsertRequest(filename)
        request.object = storage.Object(contentLanguage='en')
        response = self.__InsertFile(filename, request=request)
        self.assertEqual(15, response.size)
        self.assertEqual('en', response.contentLanguage)

    def testAutoUpload(self):
        filename = 'ten_meg_file'
        size = 10 << 20
        self.__ResetUpload(size)
        request = self.__InsertRequest(filename)
        response = self.__InsertFile(filename, request=request)
        self.assertEqual(size, response.size)

    def testStreamMedia(self):
        filename = 'ten_meg_file'
        size = 10 << 20
        self.__ResetUpload(size, auto_transfer=False)
        self.__upload.strategy = 'resumable'
        self.__upload.total_size = size
        request = self.__InsertRequest(filename)
        initial_response = self.__client.objects.Insert(request, upload=self.__upload)
        self.assertIsNotNone(initial_response)
        self.assertEqual(0, self.__buffer.tell())
        self.__upload.StreamMedia()
        self.assertEqual(size, self.__buffer.tell())

    def testBreakAndResumeUpload(self):
        filename = 'ten_meg_file_' + ''.join(random.sample(string.ascii_letters, 5))
        size = 10 << 20
        self.__ResetUpload(size, auto_transfer=False)
        self.__upload.strategy = 'resumable'
        self.__upload.total_size = size
        request = self.__InsertRequest(filename)
        initial_response = self.__client.objects.Insert(request, upload=self.__upload)
        self.assertIsNotNone(initial_response)
        self.assertEqual(0, self.__buffer.tell())
        upload_data = json.dumps(self.__upload.serialization_data)
        second_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
        second_upload_attempt._Upload__SendChunk(0)
        self.assertEqual(second_upload_attempt.chunksize, self.__buffer.tell())
        final_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
        final_upload_attempt.StreamInChunks()
        self.assertEqual(size, self.__buffer.tell())
        object_info = self.__client.objects.Get(self.__GetRequest(filename))
        self.assertEqual(size, object_info.size)
        completed_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
        self.assertTrue(completed_upload_attempt.complete)
        completed_upload_attempt.StreamInChunks()
        object_info = self.__client.objects.Get(self.__GetRequest(filename))
        self.assertEqual(size, object_info.size)
import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
class DownloadsTest(unittest.TestCase):
    _DEFAULT_BUCKET = 'apitools'
    _TESTDATA_PREFIX = 'testdata'

    def setUp(self):
        self.__client = _GetClient()
        self.__ResetDownload()

    def __ResetDownload(self, auto_transfer=False):
        self.__buffer = six.StringIO()
        self.__download = storage.Download.FromStream(self.__buffer, auto_transfer=auto_transfer)

    def __GetTestdataFileContents(self, filename):
        file_path = os.path.join(os.path.dirname(__file__), self._TESTDATA_PREFIX, filename)
        file_contents = open(file_path).read()
        self.assertIsNotNone(file_contents, msg='Could not read file %s' % filename)
        return file_contents

    @classmethod
    def __GetRequest(cls, filename):
        object_name = os.path.join(cls._TESTDATA_PREFIX, filename)
        return storage.StorageObjectsGetRequest(bucket=cls._DEFAULT_BUCKET, object=object_name)

    def __GetFile(self, request):
        response = self.__client.objects.Get(request, download=self.__download)
        self.assertIsNone(response, msg='Unexpected nonempty response for file download: %s' % response)

    def __GetAndStream(self, request):
        self.__GetFile(request)
        self.__download.StreamInChunks()

    def testZeroBytes(self):
        request = self.__GetRequest('zero_byte_file')
        self.__GetAndStream(request)
        self.assertEqual(0, self.__buffer.tell())

    def testObjectDoesNotExist(self):
        self.__ResetDownload(auto_transfer=True)
        with self.assertRaises(exceptions.HttpError):
            self.__GetFile(self.__GetRequest('nonexistent_file'))

    def testAutoTransfer(self):
        self.__ResetDownload(auto_transfer=True)
        self.__GetFile(self.__GetRequest('fifteen_byte_file'))
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        self.assertEqual(15, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents, self.__buffer.read())

    def testFilenameWithSpaces(self):
        self.__ResetDownload(auto_transfer=True)
        self.__GetFile(self.__GetRequest('filename with spaces'))
        file_contents = self.__GetTestdataFileContents('filename_with_spaces')
        self.assertEqual(15, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents, self.__buffer.read())

    def testGetRange(self):
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        self.__GetFile(self.__GetRequest('fifteen_byte_file'))
        self.__download.GetRange(5, 10)
        self.assertEqual(6, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents[5:11], self.__buffer.read())

    def testGetRangeWithNegativeStart(self):
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        self.__GetFile(self.__GetRequest('fifteen_byte_file'))
        self.__download.GetRange(-3)
        self.assertEqual(3, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents[-3:], self.__buffer.read())

    def testGetRangeWithPositiveStart(self):
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        self.__GetFile(self.__GetRequest('fifteen_byte_file'))
        self.__download.GetRange(2)
        self.assertEqual(13, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents[2:15], self.__buffer.read())

    def testSmallChunksizes(self):
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        request = self.__GetRequest('fifteen_byte_file')
        for chunksize in (2, 3, 15, 100):
            self.__ResetDownload()
            self.__download.chunksize = chunksize
            self.__GetAndStream(request)
            self.assertEqual(15, self.__buffer.tell())
            self.__buffer.seek(0)
            self.assertEqual(file_contents, self.__buffer.read(15))

    def testLargeFileChunksizes(self):
        request = self.__GetRequest('thirty_meg_file')
        for chunksize in (1048576, 40 * 1048576):
            self.__ResetDownload()
            self.__download.chunksize = chunksize
            self.__GetAndStream(request)
            self.__buffer.seek(0)

    def testAutoGzipObject(self):
        request = storage.StorageObjectsGetRequest(bucket='ottenl-gzip', object='50K.txt')
        self.__GetFile(request)
        self.assertEqual(0, self.__buffer.tell())
        self.__download.StreamInChunks()
        self.assertEqual(50000, self.__buffer.tell())
        self.__ResetDownload(auto_transfer=True)
        self.__GetFile(request)
        self.assertEqual(50000, self.__buffer.tell())

    def testSmallGzipObject(self):
        request = self.__GetRequest('zero-gzipd.html')
        self.__GetFile(request)
        self.assertEqual(0, self.__buffer.tell())
        additional_headers = {'accept-encoding': 'gzip, deflate'}
        self.__download.StreamInChunks(additional_headers=additional_headers)
        self.assertEqual(0, self.__buffer.tell())

    def testSerializedDownload(self):

        def _ProgressCallback(unused_response, download_object):
            print('Progress %s' % download_object.progress)
        file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
        object_name = os.path.join(self._TESTDATA_PREFIX, 'fifteen_byte_file')
        request = storage.StorageObjectsGetRequest(bucket=self._DEFAULT_BUCKET, object=object_name)
        response = self.__client.objects.Get(request)
        self.__buffer = six.StringIO()
        download_data = json.dumps({'auto_transfer': False, 'progress': 0, 'total_size': response.size, 'url': response.mediaLink})
        self.__download = storage.Download.FromData(self.__buffer, download_data, http=self.__client.http)
        self.__download.StreamInChunks(callback=_ProgressCallback)
        self.assertEqual(15, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents, self.__buffer.read(15))
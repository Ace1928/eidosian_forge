import os
import unittest
from boto.s3.keyfile import KeyFile
from tests.integration.s3.mock_storage_service import MockConnection
from tests.integration.s3.mock_storage_service import MockBucket
class KeyfileTest(unittest.TestCase):

    def setUp(self):
        service_connection = MockConnection()
        self.contents = '0123456789'
        bucket = MockBucket(service_connection, 'mybucket')
        key = bucket.new_key('mykey')
        key.set_contents_from_string(self.contents)
        self.keyfile = KeyFile(key)

    def tearDown(self):
        self.keyfile.close()

    def testReadFull(self):
        self.assertEqual(self.keyfile.read(len(self.contents)), self.contents)

    def testReadPartial(self):
        self.assertEqual(self.keyfile.read(5), self.contents[:5])
        self.assertEqual(self.keyfile.read(5), self.contents[5:])

    def testTell(self):
        self.assertEqual(self.keyfile.tell(), 0)
        self.keyfile.read(4)
        self.assertEqual(self.keyfile.tell(), 4)
        self.keyfile.read(6)
        self.assertEqual(self.keyfile.tell(), 10)
        self.keyfile.close()
        try:
            self.keyfile.tell()
        except ValueError as e:
            self.assertEqual(str(e), 'I/O operation on closed file')

    def testSeek(self):
        self.assertEqual(self.keyfile.read(4), self.contents[:4])
        self.keyfile.seek(0)
        self.assertEqual(self.keyfile.read(4), self.contents[:4])
        self.keyfile.seek(5)
        self.assertEqual(self.keyfile.read(5), self.contents[5:])
        try:
            self.keyfile.seek(-5)
        except IOError as e:
            self.assertEqual(str(e), 'Invalid argument')
        self.keyfile.read(10)
        self.assertEqual(self.keyfile.read(20), '')
        self.keyfile.seek(50)
        self.assertEqual(self.keyfile.tell(), 50)
        self.assertEqual(self.keyfile.read(1), '')

    def testSeekEnd(self):
        self.assertEqual(self.keyfile.read(4), self.contents[:4])
        self.keyfile.seek(0, os.SEEK_END)
        self.assertEqual(self.keyfile.read(1), '')
        self.keyfile.seek(-1, os.SEEK_END)
        self.assertEqual(self.keyfile.tell(), 9)
        self.assertEqual(self.keyfile.read(1), '9')
        try:
            self.keyfile.seek(-100, os.SEEK_END)
        except IOError as e:
            self.assertEqual(str(e), 'Invalid argument')

    def testSeekCur(self):
        self.assertEqual(self.keyfile.read(1), self.contents[0])
        self.keyfile.seek(1, os.SEEK_CUR)
        self.assertEqual(self.keyfile.tell(), 2)
        self.assertEqual(self.keyfile.read(4), self.contents[2:6])

    def testSetEtag(self):
        self.keyfile.key.data = b'test'
        self.keyfile.key.set_etag()
        self.assertEqual(self.keyfile.key.etag, '098f6bcd4621d373cade4e832627b4f6')
        self.keyfile.key.etag = None
        self.keyfile.key.data = 'test'
        self.keyfile.key.set_etag()
        self.assertEqual(self.keyfile.key.etag, '098f6bcd4621d373cade4e832627b4f6')
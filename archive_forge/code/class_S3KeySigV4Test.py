from tests.unit import unittest
import time
import random
import boto.s3
from boto.compat import six, StringIO, urllib
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3ResponseError
class S3KeySigV4Test(unittest.TestCase):

    def setUp(self):
        self.conn = boto.s3.connect_to_region('eu-central-1')
        self.bucket_name = 'boto-sigv4-key-%d' % int(time.time())
        self.bucket = self.conn.create_bucket(self.bucket_name, location='eu-central-1')

    def tearDown(self):
        for key in self.bucket:
            key.delete()
        self.bucket.delete()

    def test_put_get_with_non_string_headers_key(self):
        k = Key(self.bucket)
        k.key = 'foobar'
        body = 'This is a test of S3'
        k.set_contents_from_string(body)
        headers = {'Content-Length': 0}
        from_s3_key = self.bucket.get_key('foobar', headers=headers)
        self.assertEqual(from_s3_key.get_contents_as_string().decode('utf-8'), body)

    def test_head_put_get_with_non_ascii_key(self):
        k = Key(self.bucket)
        k.key = u'pt-Olá_ch-你好_ko-안녕_ru-Здравствуйте%20,.<>~`!@#$%^&()_-+=\'"'
        body = 'This is a test of S3'
        k.set_contents_from_string(body)
        from_s3_key = self.bucket.get_key(k.key, validate=True)
        self.assertEqual(from_s3_key.get_contents_as_string().decode('utf-8'), body)
        keys = self.bucket.get_all_keys(prefix=k.key, max_keys=1)
        self.assertEqual(1, len(keys))
import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
class S3MultiPartUploadSigV4Test(unittest.TestCase):
    s3 = True

    def setUp(self):
        self.env_patch = mock.patch('os.environ', {'S3_USE_SIGV4': True})
        self.env_patch.start()
        self.conn = boto.s3.connect_to_region('us-west-2')
        self.bucket_name = 'multipart-%d' % int(time.time())
        self.bucket = self.conn.create_bucket(self.bucket_name, location='us-west-2')

    def tearDown(self):
        for key in self.bucket:
            key.delete()
        self.bucket.delete()
        self.env_patch.stop()

    def test_initiate_multipart(self):
        key_name = 'multipart'
        multipart_upload = self.bucket.initiate_multipart_upload(key_name)
        multipart_uploads = self.bucket.get_all_multipart_uploads()
        for upload in multipart_uploads:
            self.assertEqual(upload.key_name, multipart_upload.key_name)
            self.assertEqual(upload.id, multipart_upload.id)
        multipart_upload.cancel_upload()

    def test_upload_part_by_size(self):
        key_name = 'k'
        contents = '01234567890123456789'
        sfp = StringIO(contents)
        mpu = self.bucket.initiate_multipart_upload(key_name)
        mpu.upload_part_from_file(sfp, part_num=1, size=5)
        mpu.upload_part_from_file(sfp, part_num=2, size=5)
        mpu.upload_part_from_file(sfp, part_num=3, size=5)
        mpu.upload_part_from_file(sfp, part_num=4, size=5)
        sfp.close()
        etags = {}
        pn = 0
        for part in mpu:
            pn += 1
            self.assertEqual(5, part.size)
            etags[pn] = part.etag
        self.assertEqual(pn, 4)
        self.assertEqual(etags[1], etags[3])
        self.assertEqual(etags[2], etags[4])
        self.assertNotEqual(etags[1], etags[2])
        mpu.cancel_upload()
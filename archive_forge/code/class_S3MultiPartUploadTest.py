import os
import unittest
import time
from boto.compat import StringIO
import mock
import boto
from boto.s3.connection import S3Connection
class S3MultiPartUploadTest(unittest.TestCase):
    s3 = True

    def setUp(self):
        self.conn = S3Connection(is_secure=False)
        self.bucket_name = 'multipart-%d' % int(time.time())
        self.bucket = self.conn.create_bucket(self.bucket_name)

    def tearDown(self):
        for key in self.bucket:
            key.delete()
        self.bucket.delete()

    def test_abort(self):
        key_name = u'テスト'
        mpu = self.bucket.initiate_multipart_upload(key_name)
        mpu.cancel_upload()

    def test_complete_ascii(self):
        key_name = 'test'
        mpu = self.bucket.initiate_multipart_upload(key_name)
        fp = StringIO('small file')
        mpu.upload_part_from_file(fp, part_num=1)
        fp.close()
        cmpu = mpu.complete_upload()
        self.assertEqual(cmpu.key_name, key_name)
        self.assertNotEqual(cmpu.etag, None)

    def test_complete_japanese(self):
        key_name = u'テスト'
        mpu = self.bucket.initiate_multipart_upload(key_name)
        fp = StringIO('small file')
        mpu.upload_part_from_file(fp, part_num=1)
        fp.close()
        cmpu = mpu.complete_upload()
        self.assertEqual(cmpu.key_name, key_name)
        self.assertNotEqual(cmpu.etag, None)

    def test_list_japanese(self):
        key_name = u'テスト'
        mpu = self.bucket.initiate_multipart_upload(key_name)
        rs = self.bucket.list_multipart_uploads()
        lmpu = next(iter(rs))
        self.assertEqual(lmpu.id, mpu.id)
        self.assertEqual(lmpu.key_name, key_name)
        lmpu.cancel_upload()

    def test_list_multipart_uploads(self):
        key_name = u'テスト'
        mpus = []
        mpus.append(self.bucket.initiate_multipart_upload(key_name))
        mpus.append(self.bucket.initiate_multipart_upload(key_name))
        rs = self.bucket.list_multipart_uploads()
        for lmpu in rs:
            ompu = mpus.pop(0)
            self.assertEqual(lmpu.key_name, ompu.key_name)
            self.assertEqual(lmpu.id, ompu.id)
        self.assertEqual(0, len(mpus))

    def test_get_all_multipart_uploads(self):
        key1 = 'a'
        key2 = 'b/c'
        mpu1 = self.bucket.initiate_multipart_upload(key1)
        mpu2 = self.bucket.initiate_multipart_upload(key2)
        rs = self.bucket.get_all_multipart_uploads(prefix='b/', delimiter='/')
        for lmpu in rs:
            self.assertEqual(lmpu.key_name, mpu2.key_name)
            self.assertEqual(lmpu.id, mpu2.id)

    def test_four_part_file(self):
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

    def test_etag_of_parts(self):
        key_name = 'etagtest'
        mpu = self.bucket.initiate_multipart_upload(key_name)
        fp = StringIO('small file')
        uparts = []
        uparts.append(mpu.upload_part_from_file(fp, part_num=1, size=5))
        uparts.append(mpu.upload_part_from_file(fp, part_num=2))
        fp.close()
        pn = 0
        for lpart in mpu:
            self.assertEqual(uparts[pn].etag, lpart.etag)
            pn += 1
        mpu.cancel_upload()
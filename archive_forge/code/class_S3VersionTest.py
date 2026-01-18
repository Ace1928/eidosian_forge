import unittest
import time
from boto.s3.connection import S3Connection
from boto.exception import S3ResponseError
from boto.s3.deletemarker import DeleteMarker
from boto.compat import six
class S3VersionTest(unittest.TestCase):

    def setUp(self):
        self.conn = S3Connection()
        self.bucket_name = 'version-%d' % int(time.time())
        self.bucket = self.conn.create_bucket(self.bucket_name)

    def tearDown(self):
        for k in self.bucket.list_versions():
            self.bucket.delete_key(k.name, version_id=k.version_id)
        self.bucket.delete()

    def test_1_versions(self):
        d = self.bucket.get_versioning_status()
        self.assertFalse('Versioning' in d)
        self.bucket.configure_versioning(versioning=True)
        d = self.bucket.get_versioning_status()
        self.assertEqual('Enabled', d['Versioning'])
        k = self.bucket.new_key('foobar')
        s1 = 'This is v1'
        k.set_contents_from_string(s1)
        v1 = k.version_id
        o1 = k.get_contents_as_string().decode('utf-8')
        self.assertEqual(s1, o1)
        s2 = 'This is v2'
        k.set_contents_from_string(s2)
        v2 = k.version_id
        k2 = self.bucket.new_key('foobar')
        o2 = k2.get_contents_as_string().decode('utf-8')
        self.assertEqual(s2, o2)
        o1 = k.get_contents_as_string(version_id=v1).decode('utf-8')
        o2 = k.get_contents_as_string(version_id=v2).decode('utf-8')
        self.assertEqual(s1, o1)
        self.assertEqual(s2, o2)
        rs = self.bucket.get_all_versions()
        self.assertEqual(v2, rs[0].version_id)
        self.assertEqual(v1, rs[1].version_id)
        rs = self.bucket.get_all_keys()
        self.assertEqual(1, len(rs))
        self.bucket.delete_key('foobar')
        rs = self.bucket.get_all_versions()
        self.assertEqual(3, len(rs))
        self.assertTrue(isinstance(rs[0], DeleteMarker))
        self.bucket.delete_key('foobar', version_id=v1)
        rs = self.bucket.get_all_versions()
        versions = [k.version_id for k in rs]
        self.assertTrue(v1 not in versions)
        self.assertTrue(v2 in versions)
        self.bucket.configure_versioning(False)
        time.sleep(3)
        d = self.bucket.get_versioning_status()
        self.assertEqual('Suspended', d['Versioning'])

    def test_latest_version(self):
        self.bucket.configure_versioning(versioning=True)
        key_name = 'key'
        kv1 = self.bucket.new_key(key_name)
        kv1.set_contents_from_string('v1')
        listed_kv1 = next(iter(self.bucket.get_all_versions()))
        self.assertEqual(listed_kv1.name, key_name)
        self.assertEqual(listed_kv1.version_id, kv1.version_id)
        self.assertEqual(listed_kv1.is_latest, True)
        kv2 = self.bucket.new_key(key_name)
        kv2.set_contents_from_string('v2')
        i = iter(self.bucket.get_all_versions())
        listed_kv2 = next(i)
        listed_kv1 = next(i)
        self.assertEqual(listed_kv2.version_id, kv2.version_id)
        self.assertEqual(listed_kv1.version_id, kv1.version_id)
        self.assertEqual(listed_kv2.is_latest, True)
        self.assertEqual(listed_kv1.is_latest, False)
        self.bucket.delete_key(key_name)
        i = iter(self.bucket.get_all_versions())
        listed_kv3 = next(i)
        listed_kv2 = next(i)
        listed_kv1 = next(i)
        self.assertNotEqual(listed_kv3.version_id, None)
        self.assertEqual(listed_kv2.version_id, kv2.version_id)
        self.assertEqual(listed_kv1.version_id, kv1.version_id)
        self.assertEqual(listed_kv3.is_latest, True)
        self.assertEqual(listed_kv2.is_latest, False)
        self.assertEqual(listed_kv1.is_latest, False)
from tests.unit import unittest
import time
import boto
from boto.s3.connection import S3Connection, Location
class StorageUriTest(unittest.TestCase):
    s3 = True

    def nuke_bucket(self, bucket):
        for key in bucket:
            key.delete()
        bucket.delete()

    def test_storage_uri_regionless(self):
        conn = S3Connection(host='s3-us-west-2.amazonaws.com')
        bucket_name = 'keytest-%d' % int(time.time())
        bucket = conn.create_bucket(bucket_name, location=Location.USWest2)
        self.addCleanup(self.nuke_bucket, bucket)
        suri = boto.storage_uri('s3://%s/test' % bucket_name)
        the_key = suri.new_key()
        the_key.key = 'Test301'
        the_key.set_contents_from_string('This should store in a different region.')
        alt_conn = boto.connect_s3(host='s3-us-west-2.amazonaws.com')
        alt_bucket = alt_conn.get_bucket(bucket_name)
        alt_key = alt_bucket.get_key('Test301')
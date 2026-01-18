import boto
from time import time
from tests.compat import unittest
class TestCloudTrail(unittest.TestCase):

    def test_cloudtrail(self):
        cloudtrail = boto.connect_cloudtrail()
        res = cloudtrail.describe_trails()
        if len(res['trailList']):
            self.fail('A trail already exists on this account!')
        iam = boto.connect_iam()
        response = iam.get_user()
        account_id = response['get_user_response']['get_user_result']['user']['user_id']
        s3 = boto.connect_s3()
        bucket_name = 'cloudtrail-integ-{0}'.format(time())
        policy = DEFAULT_S3_POLICY.replace('<BucketName>', bucket_name).replace('<CustomerAccountID>', account_id).replace('<Prefix>/', '')
        b = s3.create_bucket(bucket_name)
        b.set_policy(policy)
        cloudtrail.create_trail(trail={'Name': 'test', 'S3BucketName': bucket_name})
        cloudtrail.update_trail(trail={'Name': 'test', 'IncludeGlobalServiceEvents': False})
        trails = cloudtrail.describe_trails()
        self.assertEqual('test', trails['trailList'][0]['Name'])
        self.assertFalse(trails['trailList'][0]['IncludeGlobalServiceEvents'])
        cloudtrail.start_logging(name='test')
        status = cloudtrail.get_trail_status(name='test')
        self.assertTrue(status['IsLogging'])
        cloudtrail.stop_logging(name='test')
        status = cloudtrail.get_trail_status(name='test')
        self.assertFalse(status['IsLogging'])
        cloudtrail.delete_trail(name='test')
        for key in b.list():
            key.delete()
        s3.delete_bucket(bucket_name)
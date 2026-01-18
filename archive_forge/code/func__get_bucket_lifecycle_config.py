from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def _get_bucket_lifecycle_config(self):
    self.set_http_response(status_code=200)
    bucket = Bucket(self.service_connection, 'mybucket')
    return bucket.get_lifecycle_config()
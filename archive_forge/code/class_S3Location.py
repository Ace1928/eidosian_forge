from datetime import datetime
from boto.compat import six
class S3Location(BaseObject):

    def __init__(self, response):
        super(S3Location, self).__init__()
        self.s3_bucket = str(response['S3Bucket'])
        self.s3_key = str(response['S3Key'])
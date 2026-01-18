import uuid
import boto
from boto.sqs.message import RawMessage
from boto.exception import SQSDecodeError
def _get_bucket_key(self, s3_url):
    bucket_name = key_name = None
    if s3_url:
        if s3_url.startswith('s3://'):
            s3_components = s3_url[5:].split('/', 1)
            bucket_name = s3_components[0]
            if len(s3_components) > 1:
                if s3_components[1]:
                    key_name = s3_components[1]
        else:
            msg = 's3_url parameter should start with s3://'
            raise SQSDecodeError(msg, self)
    return (bucket_name, key_name)
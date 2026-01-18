import uuid
import base64
import time
from boto.compat import six, json
from boto.cloudfront.identity import OriginAccessIdentity
from boto.cloudfront.object import Object, StreamingObject
from boto.cloudfront.signers import ActiveTrustedSigners, TrustedSigners
from boto.cloudfront.logging import LoggingInfo
from boto.cloudfront.origin import S3Origin, CustomOrigin
from boto.s3.acl import ACL
def _get_bucket(self):
    if isinstance(self.config.origin, S3Origin):
        if not self._bucket:
            bucket_dns_name = self.config.origin.dns_name
            bucket_name = bucket_dns_name.replace('.s3.amazonaws.com', '')
            from boto.s3.connection import S3Connection
            s3 = S3Connection(self.connection.aws_access_key_id, self.connection.aws_secret_access_key, proxy=self.connection.proxy, proxy_port=self.connection.proxy_port, proxy_user=self.connection.proxy_user, proxy_pass=self.connection.proxy_pass)
            self._bucket = s3.get_bucket(bucket_name)
            self._bucket.distribution = self
            self._bucket.set_key_class(self._object_class)
        return self._bucket
    else:
        raise NotImplementedError('Unable to get_objects on CustomOrigin')
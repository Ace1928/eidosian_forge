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
@staticmethod
def _custom_policy(resource, expires=None, valid_after=None, ip_address=None):
    """
        Creates a custom policy string based on the supplied parameters.
        """
    condition = {}
    if not expires:
        expires = int(time.time()) + 86400
    condition['DateLessThan'] = {'AWS:EpochTime': expires}
    if valid_after:
        condition['DateGreaterThan'] = {'AWS:EpochTime': valid_after}
    if ip_address:
        if '/' not in ip_address:
            ip_address += '/32'
        condition['IpAddress'] = {'AWS:SourceIp': ip_address}
    policy = {'Statement': [{'Resource': resource, 'Condition': condition}]}
    return json.dumps(policy, separators=(',', ':'))
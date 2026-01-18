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
def _create_signing_params(self, url, keypair_id, expire_time=None, valid_after_time=None, ip_address=None, policy_url=None, private_key_file=None, private_key_string=None):
    """
        Creates the required URL parameters for a signed URL.
        """
    params = {}
    if expire_time and (not valid_after_time) and (not ip_address) and (not policy_url):
        policy = self._canned_policy(url, expire_time)
        params['Expires'] = str(expire_time)
    else:
        if policy_url is None:
            policy_url = url
        policy = self._custom_policy(policy_url, expires=expire_time, valid_after=valid_after_time, ip_address=ip_address)
        encoded_policy = self._url_base64_encode(policy)
        params['Policy'] = encoded_policy
    signature = self._sign_string(policy, private_key_file, private_key_string)
    encoded_signature = self._url_base64_encode(signature)
    params['Signature'] = encoded_signature
    params['Key-Pair-Id'] = keypair_id
    return params
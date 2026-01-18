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
def _sign_string(message, private_key_file=None, private_key_string=None):
    """
        Signs a string for use with Amazon CloudFront.
        Requires the rsa library be installed.
        """
    try:
        import rsa
    except ImportError:
        raise NotImplementedError('Boto depends on the python rsa library to generate signed URLs for CloudFront')
    if private_key_file and private_key_string:
        raise ValueError('Only specify the private_key_file or the private_key_string not both')
    if not private_key_file and (not private_key_string):
        raise ValueError('You must specify one of private_key_file or private_key_string')
    if private_key_string is None:
        if isinstance(private_key_file, six.string_types):
            with open(private_key_file, 'r') as file_handle:
                private_key_string = file_handle.read()
        else:
            private_key_string = private_key_file.read()
    private_key = rsa.PrivateKey.load_pkcs1(private_key_string)
    signature = rsa.sign(str(message), private_key, 'SHA-1')
    return signature
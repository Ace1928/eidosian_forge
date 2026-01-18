import os
from boto.compat import six
from datetime import datetime
import boto
from boto import config
from boto.compat import expanduser
from boto.pyami.config import Config
from boto.exception import InvalidInstanceMetadataError
from boto.gs.acl import ACL
from boto.gs.acl import CannedACLStrings as CannedGSACLStrings
from boto.s3.acl import CannedACLStrings as CannedS3ACLStrings
from boto.s3.acl import Policy
def _convert_key_to_str(self, key):
    if isinstance(key, six.text_type):
        return str(key)
    return key
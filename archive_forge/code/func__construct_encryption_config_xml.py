from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import xml.sax
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import GSResponseError
from boto.exception import InvalidAclError
from boto.gs.acl import ACL, CannedACLStrings
from boto.gs.acl import SupportedPermissions as GSPermissions
from boto.gs.bucketlistresultset import VersionedBucketListResultSet
from boto.gs.cors import Cors
from boto.gs.encryptionconfig import EncryptionConfig
from boto.gs.lifecycle import LifecycleConfig
from boto.gs.key import Key as GSKey
from boto.s3.acl import Policy
from boto.s3.bucket import Bucket as S3Bucket
from boto.utils import get_utf8able_str
from boto.compat import quote
from boto.compat import six
def _construct_encryption_config_xml(self, default_kms_key_name=None):
    """Creates an XML document for setting a bucket's EncryptionConfig.

        This method is internal as it's only here for testing purposes. As
        managing Cloud KMS resources for testing is complex, we settle for
        testing that we're creating correctly-formed XML for setting a bucket's
        encryption configuration.

        :param str default_kms_key_name: A string containing a fully-qualified
            Cloud KMS key name.
        :rtype: str
        """
    if default_kms_key_name:
        default_kms_key_name_frag = self.EncryptionConfigDefaultKeyNameFragment % default_kms_key_name
    else:
        default_kms_key_name_frag = ''
    return self.EncryptionConfigBody % default_kms_key_name_frag
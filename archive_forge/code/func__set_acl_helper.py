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
def _set_acl_helper(self, acl_or_str, key_name, headers, query_args, generation, if_generation, if_metageneration, canned=False):
    """Provides common functionality for set_acl, set_xml_acl,
        set_canned_acl, set_def_acl, set_def_xml_acl, and
        set_def_canned_acl()."""
    headers = headers or {}
    data = ''
    if canned:
        headers[self.connection.provider.acl_header] = acl_or_str
    else:
        data = acl_or_str
    if generation:
        query_args += '&generation=%s' % generation
    if if_metageneration is not None and if_generation is None:
        raise ValueError('Received if_metageneration argument with no if_generation argument. A metageneration has no meaning without a content generation.')
    if not key_name and (if_generation or if_metageneration):
        raise ValueError('Received if_generation or if_metageneration parameter while setting the ACL of a bucket.')
    if if_generation is not None:
        headers['x-goog-if-generation-match'] = str(if_generation)
    if if_metageneration is not None:
        headers['x-goog-if-metageneration-match'] = str(if_metageneration)
    response = self.connection.make_request('PUT', self.name, key_name, data=data, headers=headers, query_args=query_args)
    body = response.read()
    if response.status != 200:
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)
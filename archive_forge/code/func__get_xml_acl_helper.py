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
def _get_xml_acl_helper(self, key_name, headers, query_args):
    """Provides common functionality for get_xml_acl and _get_acl_helper."""
    response = self.connection.make_request('GET', self.name, key_name, query_args=query_args, headers=headers)
    body = response.read()
    if response.status != 200:
        if response.status == 403:
            match = ERROR_DETAILS_REGEX.search(body)
            details = match.group('details') if match else None
            if details:
                details = '<Details>%s. Note that Full Control access is required to access ACLs.</Details>' % details
                if six.PY3:
                    details = details.encode('utf-8')
                body = re.sub(ERROR_DETAILS_REGEX, details, body)
        raise self.connection.provider.storage_response_error(response.status, response.reason, body)
    return body
from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
def _delete_key_internal(self, key_name, headers=None, version_id=None, mfa_token=None, query_args_l=None):
    query_args_l = query_args_l or []
    provider = self.connection.provider
    if version_id:
        query_args_l.append('versionId=%s' % version_id)
    query_args = '&'.join(query_args_l) or None
    if mfa_token:
        if not headers:
            headers = {}
        headers[provider.mfa_header] = ' '.join(mfa_token)
    response = self.connection.make_request('DELETE', self.name, key_name, headers=headers, query_args=query_args)
    body = response.read()
    if response.status != 204:
        raise provider.storage_response_error(response.status, response.reason, body)
    else:
        k = self.key_class(self)
        k.name = key_name
        k.handle_version_headers(response)
        k.handle_addl_headers(response.getheaders())
        return k
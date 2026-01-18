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
def _get_all_query_args(self, params, initial_query_string=''):
    pairs = []
    if initial_query_string:
        pairs.append(initial_query_string)
    for key, value in sorted(params.items(), key=lambda x: x[0]):
        if value is None:
            continue
        key = key.replace('_', '-')
        if key == 'maxkeys':
            key = 'max-keys'
        if not isinstance(value, six.string_types + (six.binary_type,)):
            value = six.text_type(value)
        if not isinstance(value, six.binary_type):
            value = value.encode('utf-8')
        if value:
            pairs.append(u'%s=%s' % (urllib.parse.quote(key), urllib.parse.quote(value)))
    return '&'.join(pairs)
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import re
import textwrap
import xml.etree.ElementTree
from xml.etree.ElementTree import ParseError as XmlParseError
import six
from apitools.base.protorpclite.util import decode_datetime
from apitools.base.py import encoding
import boto
from boto.gs.acl import ACL
from boto.gs.acl import ALL_AUTHENTICATED_USERS
from boto.gs.acl import ALL_USERS
from boto.gs.acl import Entries
from boto.gs.acl import Entry
from boto.gs.acl import GROUP_BY_DOMAIN
from boto.gs.acl import GROUP_BY_EMAIL
from boto.gs.acl import GROUP_BY_ID
from boto.gs.acl import USER_BY_EMAIL
from boto.gs.acl import USER_BY_ID
from boto.s3.tagging import Tags
from boto.s3.tagging import TagSet
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BucketNotFoundException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import Preconditions
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
def CreateNotFoundExceptionForObjectWrite(dst_provider, dst_bucket_name, src_provider=None, src_bucket_name=None, src_object_name=None, src_generation=None):
    """Creates a NotFoundException for an object upload or copy.

  This is necessary because 404s don't necessarily specify which resource
  does not exist.

  Args:
    dst_provider: String abbreviation of destination provider, e.g., 'gs'.
    dst_bucket_name: Destination bucket name for the write operation.
    src_provider: String abbreviation of source provider, i.e. 'gs', if any.
    src_bucket_name: Source bucket name, if any (for the copy case).
    src_object_name: Source object name, if any (for the copy case).
    src_generation: Source object generation, if any (for the copy case).

  Returns:
    NotFoundException with appropriate message.
  """
    dst_url_string = '%s://%s' % (dst_provider, dst_bucket_name)
    if src_bucket_name and src_object_name:
        src_url_string = '%s://%s/%s' % (src_provider, src_bucket_name, src_object_name)
        if src_generation:
            src_url_string += '#%s' % str(src_generation)
        return NotFoundException('The source object %s or the destination bucket %s does not exist.' % (src_url_string, dst_url_string))
    return NotFoundException('The destination bucket %s does not exist or the write to the destination must be restarted' % dst_url_string)
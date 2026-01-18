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
def AddS3MarkerAclToObjectMetadata(object_metadata, acl_text):
    """Adds a GUID-marked S3 ACL to the object metadata.

  Args:
    object_metadata: Object metadata to add the acl to.
    acl_text: S3 ACL text to add.
  """
    if not object_metadata.metadata:
        object_metadata.metadata = apitools_messages.Object.MetadataValue()
    if not object_metadata.metadata.additionalProperties:
        object_metadata.metadata.additionalProperties = []
    object_metadata.metadata.additionalProperties.append(apitools_messages.Object.MetadataValue.AdditionalProperty(key=S3_ACL_MARKER_GUID, value=acl_text))
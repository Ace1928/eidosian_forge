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
def CopyCustomMetadata(src_obj_metadata, dst_obj_metadata, override=False):
    """Copies custom metadata from src_obj_metadata to dst_obj_metadata.

  Args:
    src_obj_metadata: Metadata from source object.
    dst_obj_metadata: Initialized metadata for destination object.
    override: If true, will overwrite metadata in destination object.
              If false, only writes metadata for values that don't already
              exist.
  """
    if src_obj_metadata.metadata and src_obj_metadata.metadata.additionalProperties:
        if not dst_obj_metadata.metadata:
            dst_obj_metadata.metadata = apitools_messages.Object.MetadataValue()
        if not dst_obj_metadata.metadata.additionalProperties:
            dst_obj_metadata.metadata.additionalProperties = []
        dst_metadata_dict = {}
        for dst_prop in dst_obj_metadata.metadata.additionalProperties:
            dst_metadata_dict[dst_prop.key] = dst_prop.value
        for src_prop in src_obj_metadata.metadata.additionalProperties:
            if src_prop.key in dst_metadata_dict:
                if override:
                    if src_prop.value is not None and (not src_prop.value):
                        dst_metadata_dict[src_prop.key] = None
                    else:
                        dst_metadata_dict[src_prop.key] = src_prop.value
            elif src_prop.value != '':
                dst_metadata_dict[src_prop.key] = src_prop.value
        dst_obj_metadata.metadata.additionalProperties = []
        for k, v in six.iteritems(dst_metadata_dict):
            dst_obj_metadata.metadata.additionalProperties.append(apitools_messages.Object.MetadataValue.AdditionalProperty(key=k, value=v))
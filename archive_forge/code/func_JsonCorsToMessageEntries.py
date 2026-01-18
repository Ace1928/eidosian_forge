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
@classmethod
def JsonCorsToMessageEntries(cls, json_cors):
    """Translates CORS JSON to an apitools message.

    Args:
      json_cors: JSON string representing CORS configuration.

    Raises:
      ArgumentException on invalid CORS JSON data.

    Returns:
      List of apitools Bucket.CorsValueListEntry. An empty list represents
      no CORS configuration.
    """
    deserialized_cors = None
    try:
        deserialized_cors = json.loads(json_cors)
    except ValueError:
        CheckForXmlConfigurationAndRaise('CORS', json_cors)
    if not isinstance(deserialized_cors, list):
        raise ArgumentException('CORS JSON should be formatted as a list containing one or more JSON objects.\nSee "gsutil help cors".')
    cors = []
    for cors_entry in deserialized_cors:
        cors.append(encoding.DictToMessage(cors_entry, apitools_messages.Bucket.CorsValueListEntry))
    return cors
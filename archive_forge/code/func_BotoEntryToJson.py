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
def BotoEntryToJson(cls, entry):
    """Converts a Boto ACL entry to a valid JSON dictionary."""
    acl_entry_json = {}
    scope_type_lower = entry.scope.type.lower()
    if scope_type_lower == ALL_USERS.lower():
        acl_entry_json['entity'] = 'allUsers'
    elif scope_type_lower == ALL_AUTHENTICATED_USERS.lower():
        acl_entry_json['entity'] = 'allAuthenticatedUsers'
    elif scope_type_lower == USER_BY_EMAIL.lower():
        acl_entry_json['entity'] = 'user-%s' % entry.scope.email_address
        acl_entry_json['email'] = entry.scope.email_address
    elif scope_type_lower == USER_BY_ID.lower():
        acl_entry_json['entity'] = 'user-%s' % entry.scope.id
        acl_entry_json['entityId'] = entry.scope.id
    elif scope_type_lower == GROUP_BY_EMAIL.lower():
        acl_entry_json['entity'] = 'group-%s' % entry.scope.email_address
        acl_entry_json['email'] = entry.scope.email_address
    elif scope_type_lower == GROUP_BY_ID.lower():
        acl_entry_json['entity'] = 'group-%s' % entry.scope.id
        acl_entry_json['entityId'] = entry.scope.id
    elif scope_type_lower == GROUP_BY_DOMAIN.lower():
        acl_entry_json['entity'] = 'domain-%s' % entry.scope.domain
        acl_entry_json['domain'] = entry.scope.domain
    else:
        raise ArgumentException('ACL contains invalid scope type: %s' % scope_type_lower)
    acl_entry_json['role'] = cls.XML_TO_JSON_ROLES[entry.permission]
    return acl_entry_json
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
def BotoEntryFromJson(cls, entry_json):
    """Converts a JSON entry into a Boto ACL entry."""
    entity = entry_json['entity']
    permission = cls.JSON_TO_XML_ROLES[entry_json['role']]
    if entity.lower() == ALL_USERS.lower():
        return Entry(type=ALL_USERS, permission=permission)
    elif entity.lower() == ALL_AUTHENTICATED_USERS.lower():
        return Entry(type=ALL_AUTHENTICATED_USERS, permission=permission)
    elif entity.startswith('project'):
        raise CommandException('XML API does not support project scopes, cannot translate ACL.')
    elif 'email' in entry_json:
        if entity.startswith('user'):
            scope_type = USER_BY_EMAIL
        elif entity.startswith('group'):
            scope_type = GROUP_BY_EMAIL
        return Entry(type=scope_type, email_address=entry_json['email'], permission=permission)
    elif 'entityId' in entry_json:
        if entity.startswith('user'):
            scope_type = USER_BY_ID
        elif entity.startswith('group'):
            scope_type = GROUP_BY_ID
        return Entry(type=scope_type, id=entry_json['entityId'], permission=permission)
    elif 'domain' in entry_json:
        if entity.startswith('domain'):
            scope_type = GROUP_BY_DOMAIN
        return Entry(type=scope_type, domain=entry_json['domain'], permission=permission)
    raise CommandException('Failed to translate JSON ACL to XML.')
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
class AclTranslation(object):
    """Functions for converting between various ACL formats.

    This class handles conversion to and from Boto ACL objects, JSON text,
    and apitools Message objects.
  """
    JSON_TO_XML_ROLES = {'READER': 'READ', 'WRITER': 'WRITE', 'OWNER': 'FULL_CONTROL'}
    XML_TO_JSON_ROLES = {'READ': 'READER', 'WRITE': 'WRITER', 'FULL_CONTROL': 'OWNER'}

    @classmethod
    def BotoAclFromJson(cls, acl_json):
        acl = ACL()
        acl.parent = None
        acl.entries = cls.BotoEntriesFromJson(acl_json, acl)
        return acl

    @classmethod
    def BotoAclFromMessage(cls, acl_message):
        acl_dicts = []
        for message in acl_message:
            if message == PRIVATE_DEFAULT_OBJ_ACL:
                break
            acl_dicts.append(encoding.MessageToDict(message))
        return cls.BotoAclFromJson(acl_dicts)

    @classmethod
    def BotoAclToJson(cls, acl):
        if hasattr(acl, 'entries'):
            return cls.BotoEntriesToJson(acl.entries)
        return []

    @classmethod
    def BotoObjectAclToMessage(cls, acl):
        for entry in cls.BotoAclToJson(acl):
            message = encoding.DictToMessage(entry, apitools_messages.ObjectAccessControl)
            message.kind = 'storage#objectAccessControl'
            yield message

    @classmethod
    def BotoBucketAclToMessage(cls, acl):
        for entry in cls.BotoAclToJson(acl):
            message = encoding.DictToMessage(entry, apitools_messages.BucketAccessControl)
            message.kind = 'storage#bucketAccessControl'
            yield message

    @classmethod
    def BotoEntriesFromJson(cls, acl_json, parent):
        entries = Entries(parent)
        entries.parent = parent
        entries.entry_list = [cls.BotoEntryFromJson(entry_json) for entry_json in acl_json]
        return entries

    @classmethod
    def BotoEntriesToJson(cls, entries):
        return [cls.BotoEntryToJson(entry) for entry in entries.entry_list]

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

    @classmethod
    def JsonToMessage(cls, json_data, message_type):
        """Converts the input JSON data into list of Object/BucketAccessControls.

    Args:
      json_data: String of JSON to convert.
      message_type: Which type of access control entries to return,
                    either ObjectAccessControl or BucketAccessControl.

    Raises:
      ArgumentException on invalid JSON data.

    Returns:
      List of ObjectAccessControl or BucketAccessControl elements.
    """
        try:
            deserialized_acl = json.loads(json_data)
            acl = []
            for acl_entry in deserialized_acl:
                acl.append(encoding.DictToMessage(acl_entry, message_type))
            return acl
        except ValueError:
            CheckForXmlConfigurationAndRaise('ACL', json_data)

    @classmethod
    def JsonFromMessage(cls, acl):
        """Strips unnecessary fields from an ACL message and returns valid JSON.

    Args:
      acl: iterable ObjectAccessControl or BucketAccessControl

    Returns:
      ACL JSON string.
    """
        serializable_acl = []
        if acl is not None:
            for acl_entry in acl:
                if acl_entry.kind == 'storage#objectAccessControl':
                    acl_entry.object = None
                    acl_entry.generation = None
                acl_entry.kind = None
                acl_entry.bucket = None
                acl_entry.id = None
                acl_entry.selfLink = None
                acl_entry.etag = None
                serializable_acl.append(encoding.MessageToDict(acl_entry))
        return json.dumps(serializable_acl, sort_keys=True, indent=2, separators=(',', ': '))
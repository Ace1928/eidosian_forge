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
def BotoLifecycleFromMessage(cls, lifecycle_message):
    """Translates an apitools message to a boto lifecycle object."""
    boto_lifecycle = boto.gs.lifecycle.LifecycleConfig()
    if lifecycle_message:
        for rule_message in lifecycle_message.rule:
            boto_rule = boto.gs.lifecycle.Rule()
            if rule_message.action and rule_message.action.type:
                if rule_message.action.type.lower() == 'delete':
                    boto_rule.action = boto.gs.lifecycle.DELETE
                elif rule_message.action.type.lower() == 'setstorageclass':
                    boto_rule.action = boto.gs.lifecycle.SET_STORAGE_CLASS
                    boto_rule.action_text = rule_message.action.storageClass
            if rule_message.condition:
                if rule_message.condition.age is not None:
                    boto_rule.conditions[boto.gs.lifecycle.AGE] = str(rule_message.condition.age)
                if rule_message.condition.createdBefore:
                    boto_rule.conditions[boto.gs.lifecycle.CREATED_BEFORE] = str(rule_message.condition.createdBefore)
                if rule_message.condition.isLive is not None:
                    boto_rule.conditions[boto.gs.lifecycle.IS_LIVE] = str(rule_message.condition.isLive).lower()
                if rule_message.condition.matchesStorageClass:
                    boto_rule.conditions[boto.gs.lifecycle.MATCHES_STORAGE_CLASS] = [str(sc) for sc in rule_message.condition.matchesStorageClass]
                if rule_message.condition.numNewerVersions is not None:
                    boto_rule.conditions[boto.gs.lifecycle.NUM_NEWER_VERSIONS] = str(rule_message.condition.numNewerVersions)
            boto_lifecycle.append(boto_rule)
    return boto_lifecycle
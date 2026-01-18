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
def BotoCorsFromMessage(cls, cors_message):
    """Translates an apitools message to a boto Cors object."""
    cors = boto.gs.cors.Cors()
    cors.cors = []
    for collection_message in cors_message:
        collection_elements = []
        if collection_message.maxAgeSeconds:
            collection_elements.append((boto.gs.cors.MAXAGESEC, str(collection_message.maxAgeSeconds)))
        if collection_message.method:
            method_elements = []
            for method in collection_message.method:
                method_elements.append((boto.gs.cors.METHOD, method))
            collection_elements.append((boto.gs.cors.METHODS, method_elements))
        if collection_message.origin:
            origin_elements = []
            for origin in collection_message.origin:
                origin_elements.append((boto.gs.cors.ORIGIN, origin))
            collection_elements.append((boto.gs.cors.ORIGINS, origin_elements))
        if collection_message.responseHeader:
            header_elements = []
            for header in collection_message.responseHeader:
                header_elements.append((boto.gs.cors.HEADER, header))
            collection_elements.append((boto.gs.cors.HEADERS, header_elements))
        cors.cors.append(collection_elements)
    return cors
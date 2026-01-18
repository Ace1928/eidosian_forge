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
def CheckForXmlConfigurationAndRaise(config_type_string, json_txt):
    """Checks a JSON parse exception for provided XML configuration."""
    try:
        xml.etree.ElementTree.fromstring(str(json_txt))
        raise ArgumentException('\n'.join(textwrap.wrap("XML {0} data provided; Google Cloud Storage {0} configuration now uses JSON format. To convert your {0}, set the desired XML ACL using 'gsutil {1} set ...' with gsutil version 3.x. Then use 'gsutil {1} get ...' with gsutil version 4 or greater to get the corresponding JSON {0}.".format(config_type_string, config_type_string.lower()))))
    except XmlParseError:
        pass
    raise ArgumentException('JSON %s data could not be loaded from: %s' % (config_type_string, json_txt))
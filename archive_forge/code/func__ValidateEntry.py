from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def _ValidateEntry(self, entry):
    if not entry.subnet:
        return MISSING_SUBNET
    try:
        ipaddr.IPNetwork(entry.subnet)
    except ValueError:
        return BAD_IPV_SUBNET % entry.subnet
    parts = entry.subnet.split('/')
    if len(parts) == 2 and (not re.match('^[0-9]+$', parts[1])):
        return BAD_PREFIX_LENGTH % entry.subnet
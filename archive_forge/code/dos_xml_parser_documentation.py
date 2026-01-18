from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
Processes XML <blacklist> nodes into BlacklistEntry objects.

    The following information is parsed out:
      subnet: The IP, in CIDR notation.
      description: (optional)
    If there are no errors, the data is loaded into a BlackListEntry object
    and added to a list. Upon error, a description of the error is added to
    a list and the method terminates.

    Args:
      node: <blacklist> XML node in dos.xml.
    
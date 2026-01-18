from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def ProcessBlacklistNode(self, node):
    """Processes XML <blacklist> nodes into BlacklistEntry objects.

    The following information is parsed out:
      subnet: The IP, in CIDR notation.
      description: (optional)
    If there are no errors, the data is loaded into a BlackListEntry object
    and added to a list. Upon error, a description of the error is added to
    a list and the method terminates.

    Args:
      node: <blacklist> XML node in dos.xml.
    """
    tag = xml_parser_utils.GetTag(node)
    if tag != 'blacklist':
        self.errors.append('Unrecognized node: <%s>' % tag)
        return
    entry = BlacklistEntry()
    entry.subnet = xml_parser_utils.GetChildNodeText(node, 'subnet')
    entry.description = xml_parser_utils.GetChildNodeText(node, 'description')
    validation = self._ValidateEntry(entry)
    if validation:
        self.errors.append(validation)
        return
    self.blacklist_entries.append(entry)
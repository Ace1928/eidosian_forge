from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from xml.etree import ElementTree
import ipaddr
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class DosXmlParser(object):
    """Provides logic for walking down XML tree and pulling data."""

    def ProcessXml(self, xml_str):
        """Parses XML string and returns object representation of relevant info.

    Args:
      xml_str: The XML string.
    Returns:
      A list of BlacklistEntry objects containing information about blacklisted
      IP's specified in the XML.
    Raises:
      AppEngineConfigException: In case of malformed XML or illegal inputs.
    """
        try:
            self.blacklist_entries = []
            self.errors = []
            xml_root = ElementTree.fromstring(xml_str)
            if xml_root.tag != 'blacklistentries':
                raise AppEngineConfigException('Root tag must be <blacklistentries>')
            for child in list(xml_root.getchildren()):
                self.ProcessBlacklistNode(child)
            if self.errors:
                raise AppEngineConfigException('\n'.join(self.errors))
            return self.blacklist_entries
        except ElementTree.ParseError:
            raise AppEngineConfigException('Bad input -- not valid XML')

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
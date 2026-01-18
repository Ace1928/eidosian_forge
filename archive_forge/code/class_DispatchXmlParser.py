from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
class DispatchXmlParser(object):
    """Provides logic for walking down XML tree and pulling data."""

    def ProcessXml(self, xml_str):
        """Parses XML string and returns object representation of relevant info.

    Args:
      xml_str: The XML string.
    Returns:
      A list of DispatchEntry objects defining how URLs are dispatched to
      modules.
    Raises:
      AppEngineConfigException: In case of malformed XML or illegal inputs.
    """
        try:
            self.dispatch_entries = []
            self.errors = []
            xml_root = ElementTree.fromstring(xml_str)
            if xml_root.tag != 'dispatch-entries':
                raise AppEngineConfigException('Root tag must be <dispatch-entries>')
            for child in list(xml_root):
                self.ProcessDispatchNode(child)
            if self.errors:
                raise AppEngineConfigException('\n'.join(self.errors))
            return self.dispatch_entries
        except ElementTree.ParseError:
            raise AppEngineConfigException('Bad input -- not valid XML')

    def ProcessDispatchNode(self, node):
        """Processes XML <dispatch> nodes into DispatchEntry objects.

    The following information is parsed out:
      url: The URL or URL pattern to route.
      module: The module to route it to.
    If there are no errors, the data is loaded into a DispatchEntry object
    and added to a list. Upon error, a description of the error is added to
    a list and the method terminates.

    Args:
      node: <dispatch> XML node in dos.xml.
    """
        tag = xml_parser_utils.GetTag(node)
        if tag != 'dispatch':
            self.errors.append('Unrecognized node: <%s>' % tag)
            return
        entry = DispatchEntry()
        entry.url = xml_parser_utils.GetChildNodeText(node, 'url')
        entry.module = xml_parser_utils.GetChildNodeText(node, 'module')
        validation = self._ValidateEntry(entry)
        if validation:
            self.errors.append(validation)
            return
        self.dispatch_entries.append(entry)

    def _ValidateEntry(self, entry):
        if not entry.url:
            return MISSING_URL
        if not entry.module:
            return MISSING_MODULE
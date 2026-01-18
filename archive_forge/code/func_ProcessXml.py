from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
from googlecloudsdk.third_party.appengine._internal import six_subset
def ProcessXml(self, xml_str):
    """Parses XML string and returns object representation of relevant info.

    Args:
      xml_str: The XML string.
    Returns:
      A list of Cron objects containing information about cron jobs from the
      XML.
    Raises:
      AppEngineConfigException: In case of malformed XML or illegal inputs.
    """
    try:
        self.crons = []
        self.errors = []
        xml_root = ElementTree.fromstring(xml_str)
        if xml_root.tag != 'cronentries':
            raise AppEngineConfigException('Root tag must be <cronentries>')
        for child in list(xml_root):
            self.ProcessCronNode(child)
        if self.errors:
            raise AppEngineConfigException('\n'.join(self.errors))
        return self.crons
    except ElementTree.ParseError:
        raise AppEngineConfigException('Bad input -- not valid XML')
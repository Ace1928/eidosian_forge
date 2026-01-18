from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def GetQueueYaml(unused_application, queue_xml_str):
    queue_xml = QueueXmlParser().ProcessXml(queue_xml_str)
    return queue_xml.ToYaml()
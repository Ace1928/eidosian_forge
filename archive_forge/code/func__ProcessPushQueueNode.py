from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def _ProcessPushQueueNode(self, node, queue):
    if xml_parser_utils.GetChild(node, 'acl') is not None:
        self.errors.append("The element <acl> is not defined for push queues; bad <queue> entry with name '%s'" % queue.name)
    for tag in PUSH_QUEUE_TAGS:
        field_name = tag.replace('-', '_')
        setattr(queue, field_name, xml_parser_utils.GetChildNodeText(node, tag))
    self._ProcessRetryParametersNode(node, queue)
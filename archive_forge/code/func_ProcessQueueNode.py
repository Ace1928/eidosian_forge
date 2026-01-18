from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.tools import xml_parser_utils
from googlecloudsdk.third_party.appengine.tools.app_engine_config_exception import AppEngineConfigException
def ProcessQueueNode(self, node):
    """Processes XML <queue> nodes into Queue objects.

    The following information is parsed out:
      name
      mode: can be either push or pull
      retry-parameters:
        task-retry-limit
    ---- push queues only ----
        task-age-limit
        min-backoff-seconds
        max-back-off-seconds
        max-doubling
      bucket-size
      max-concurrent-requests
      rate: how often tasks are processed on this queue.
      target: version of application on which tasks on this queue will be
        invoked.
    ---- pull queues only ----
      acl: access control list - lists user and writer email addresses.

    Args:
      node: Current <queue> XML node being processed.
    """
    name = xml_parser_utils.GetChildNodeText(node, 'name')
    if not name:
        self.errors.append('Must specify a name for each <queue> entry')
        return
    mode = xml_parser_utils.GetChildNodeText(node, 'mode', 'push')
    if mode not in ('push', 'pull'):
        self.errors.append(BAD_MODE_ERROR_MESSAGE % (mode, name))
        return
    if mode == 'pull':
        queue = PullQueue()
        queue.name = name
        self._ProcessPullQueueNode(node, queue)
    else:
        queue = PushQueue()
        queue.name = name
        self._ProcessPushQueueNode(node, queue)
    self.queue_xml.queues.append(queue)
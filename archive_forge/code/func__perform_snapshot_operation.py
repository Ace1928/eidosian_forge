import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _perform_snapshot_operation(self, node, operation, xml_data, headers):
    res = self.connection.request('{}/action/{}'.format(get_url_path(node.id), operation), data=ET.tostring(xml_data) if xml_data is not None else None, method='POST', headers=headers)
    self._wait_for_task_completion(res.object.get('href'))
    res = self.connection.request(get_url_path(node.id))
    return self._to_node(res.object)
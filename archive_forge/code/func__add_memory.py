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
def _add_memory(self, parent):
    mem_item = ET.SubElement(parent, 'Item', {'xmlns': 'http://schemas.dmtf.org/ovf/envelope/1'})
    self._add_instance_id(mem_item, '2')
    self._add_resource_type(mem_item, '4')
    self._add_virtual_quantity(mem_item, self.memory)
    return mem_item
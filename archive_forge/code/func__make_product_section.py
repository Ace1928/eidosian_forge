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
def _make_product_section(self, parent):
    prod_section = ET.SubElement(parent, 'ProductSection', {'xmlns:q1': 'http://www.vmware.com/vcloud/v0.8', 'xmlns:ovf': 'http://schemas.dmtf.org/ovf/envelope/1'})
    if self.password:
        self._add_property(prod_section, 'password', self.password)
    if self.row:
        self._add_property(prod_section, 'row', self.row)
    if self.group:
        self._add_property(prod_section, 'group', self.group)
    return prod_section
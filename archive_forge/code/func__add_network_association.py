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
def _add_network_association(self, parent):
    if self.vm_network is None:
        parent.set('networkName', self.network.get('name'))
    else:
        parent.set('networkName', self.vm_network)
    configuration = ET.SubElement(parent, 'Configuration')
    ET.SubElement(configuration, 'ParentNetwork', {'href': self.network.get('href')})
    if self.vm_fence is None:
        fencemode = self.network.find(fixxpath(self.network, 'Configuration/FenceMode')).text
    else:
        fencemode = self.vm_fence
    ET.SubElement(configuration, 'FenceMode').text = fencemode
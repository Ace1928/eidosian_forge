import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisReservedIpAddress:
    """
    NTTCIS Rerverse IPv4 address
    """

    def __init__(self, datacenter_id, exclusive, vlan_id, ip, description=None):
        self.datacenter_id = datacenter_id
        self.exclusive = exclusive
        self.vlan_id = vlan_id
        self.ip = ip
        self.description = description

    def __repr__(self):
        return '<NttCisReservedIpAddress datacenterId=%s, exclusiven=%s, vlanId=%s, ipAddress=%s, description=-%s' % (self.datacenter_id, self.exclusive, self.vlan_id, self.ip, self.description)
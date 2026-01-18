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
class NttCisFirewallAddress:
    """
    The source or destination model in a firewall rule
    9/4/18: Editing Class to use with ex_create_firewall_rtule method.
    Will haved to circle back and test for any other uses.
    """

    def __init__(self, any_ip=None, ip_address=None, ip_prefix_size=None, port_begin=None, port_end=None, address_list_id=None, port_list_id=None):
        """
        :param any_ip: used to set ip address to "ANY"
        :param ip_address: Optional, an ip address of either IPv4 decimal
                           notation or an IPv6 address
        :type ``str``

        :param ip_prefix_size: An integer denoting prefix size.
        :type ``int``

        :param port_begin: integer for an individual port or start of a list
                           of ports if not using a port list
        :type ``int``

        :param port_end: integer required if using a list of ports
                         (NOT a port list but a list starting with port begin)
        :type  ``int``

        :param address_list_id: An id identifying an address list
        :type ``str``

        :param port_list_id:  An id identifying a port list
        :type ``str``
        """
        self.any_ip = any_ip
        self.ip_address = ip_address
        self.ip_prefix_size = ip_prefix_size
        self.port_list_id = port_list_id
        self.port_begin = port_begin
        self.port_end = port_end
        self.address_list_id = address_list_id
        self.port_list_id = port_list_id

    def __repr__(self):
        return '<NttCisFirewallAddress: any_ip=%s, ip_address=%s, ip_prefix_size=%s, port_begin=%s, port_end=%s, address_list_id=%s, port_list_id=%s>' % (self.any_ip, self.ip_address, self.ip_prefix_size, self.port_begin, self.port_end, self.address_list_id, self.port_list_id)
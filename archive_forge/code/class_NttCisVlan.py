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
class NttCisVlan:
    """
    NTTCIS VLAN.
    """

    def __init__(self, id, name, description, location, network_domain, status, private_ipv4_range_address, private_ipv4_range_size, ipv6_range_address, ipv6_range_size, ipv4_gateway, ipv6_gateway):
        """
        Initialize an instance of ``DimensionDataVlan``

        :param id: The ID of the VLAN
        :type  id: ``str``

        :param name: The name of the VLAN
        :type  name: ``str``

        :param description: Plan text description of the VLAN
        :type  description: ``str``

        :param location: The location (data center) of the VLAN
        :type  location: ``NodeLocation``

        :param network_domain: The Network Domain that owns this VLAN
        :type  network_domain: :class:`DimensionDataNetworkDomain`

        :param status: The status of the VLAN
        :type  status: :class:`DimensionDataStatus`

        :param private_ipv4_range_address: The host address of the VLAN
                                            IP space
        :type  private_ipv4_range_address: ``str``

        :param private_ipv4_range_size: The size (e.g. '24') of the VLAN
                                            as a CIDR range size
        :type  private_ipv4_range_size: ``int``

        :param ipv6_range_address: The host address of the VLAN
                                            IP space
        :type  ipv6_range_address: ``str``

        :param ipv6_range_size: The size (e.g. '32') of the VLAN
                                            as a CIDR range size
        :type  ipv6_range_size: ``int``

        :param ipv4_gateway: The IPv4 default gateway address
        :type  ipv4_gateway: ``str``

        :param ipv6_gateway: The IPv6 default gateway address
        :type  ipv6_gateway: ``str``
        """
        self.id = str(id)
        self.name = name
        self.location = location
        self.description = description
        self.network_domain = network_domain
        self.status = status
        self.private_ipv4_range_address = private_ipv4_range_address
        self.private_ipv4_range_size = private_ipv4_range_size
        self.ipv6_range_address = ipv6_range_address
        self.ipv6_range_size = ipv6_range_size
        self.ipv4_gateway = ipv4_gateway
        self.ipv6_gateway = ipv6_gateway

    def __repr__(self):
        return '<NttCisVlan: id=%s, name=%s, description=%s, location=%s, status=%s>' % (self.id, self.name, self.description, self.location, self.status)
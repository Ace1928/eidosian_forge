import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_firewall_address(self, element):
    ip = element.find(fixxpath('ip', TYPES_URN))
    port = element.find(fixxpath('port', TYPES_URN))
    port_list = element.find(fixxpath('portList', TYPES_URN))
    address_list = element.find(fixxpath('ipAddressList', TYPES_URN))
    if address_list is None:
        return NttCisFirewallAddress(any_ip=ip.get('address') == 'ANY', ip_address=ip.get('address'), ip_prefix_size=ip.get('prefixSize'), port_begin=port.get('begin') if port is not None else None, port_end=port.get('end') if port is not None else None, port_list_id=port_list.get('id', None) if port_list is not None else None, address_list_id=address_list.get('id') if address_list is not None else None)
    else:
        return NttCisFirewallAddress(any_ip=False, ip_address=None, ip_prefix_size=None, port_begin=None, port_end=None, port_list_id=port_list.get('id', None) if port_list is not None else None, address_list_id=address_list.get('id') if address_list is not None else None)
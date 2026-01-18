import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def _parse_ips_from_nic(self, nic):
    """
        Parse private and public IP addresses from the provided network
        interface object.

        :param nic: NIC object.
        :type nic: ``dict``

        :return: (public_ips, private_ips) tuple.
        :rtype: ``tuple``
        """
    public_ips, private_ips = ([], [])
    ipv4_conf = nic['ip_v4_conf']
    ipv6_conf = nic['ip_v6_conf']
    ip_v4 = ipv4_conf['ip'] if ipv4_conf else None
    ip_v6 = ipv6_conf['ip'] if ipv6_conf else None
    ipv4 = ip_v4['uuid'] if ip_v4 else None
    ipv6 = ip_v4['uuid'] if ip_v6 else None
    ips = []
    if ipv4:
        ips.append(ipv4)
    if ipv6:
        ips.append(ipv6)
    runtime = nic['runtime']
    ip_v4 = runtime['ip_v4'] if nic['runtime'] else None
    ip_v6 = runtime['ip_v6'] if nic['runtime'] else None
    ipv4 = ip_v4['uuid'] if ip_v4 else None
    ipv6 = ip_v4['uuid'] if ip_v6 else None
    if ipv4:
        ips.append(ipv4)
    if ipv6:
        ips.append(ipv6)
    ips = set(ips)
    for ip in ips:
        if is_private_subnet(ip):
            private_ips.append(ip)
        else:
            public_ips.append(ip)
    return (public_ips, private_ips)
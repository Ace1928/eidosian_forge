import os
import socket
import struct
from six.moves.urllib.parse import urlparse
def _is_no_proxy_host(hostname, no_proxy):
    if not no_proxy:
        v = os.environ.get('no_proxy', '').replace(' ', '')
        no_proxy = v.split(',')
    if not no_proxy:
        no_proxy = DEFAULT_NO_PROXY_HOST
    if hostname in no_proxy:
        return True
    elif _is_ip_address(hostname):
        return any([_is_address_in_network(hostname, subnet) for subnet in no_proxy if _is_subnet_address(subnet)])
    return False
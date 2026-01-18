import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def _get_my_ipv6_address():
    """Figure out the best IPv6 address
    """
    LOCALHOST = '::1'
    gtw = netifaces.gateways()
    try:
        interface = gtw['default'][netifaces.AF_INET6][1]
    except (KeyError, IndexError):
        LOG.info('Could not determine default network interface, using %s for IPv6 address', LOCALHOST)
        return LOCALHOST
    try:
        return netifaces.ifaddresses(interface)[netifaces.AF_INET6][0]['addr']
    except (KeyError, IndexError):
        LOG.info('Could not determine IPv6 address for interface %(interface)s, using %(address)s', {'interface': interface, 'address': LOCALHOST})
    except Exception as e:
        LOG.info('Could not determine IPv6 address for interface %(interface)s: %(error)s', {'interface': interface, 'error': e})
    return LOCALHOST
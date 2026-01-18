import logging
import os
import re
import socket
from urllib import parse
import netaddr
from netaddr.core import INET_PTON
import netifaces
from oslo_utils._i18n import _
def _get_my_ipv4_address():
    """Figure out the best ipv4
    """
    LOCALHOST = '127.0.0.1'
    gtw = netifaces.gateways()
    try:
        interface = gtw['default'][netifaces.AF_INET][1]
    except (KeyError, IndexError):
        LOG.info('Could not determine default network interface, using 127.0.0.1 for IPv4 address')
        return LOCALHOST
    try:
        return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
    except (KeyError, IndexError):
        LOG.info('Could not determine IPv4 address for interface %s, using 127.0.0.1', interface)
    except Exception as e:
        LOG.info('Could not determine IPv4 address for interface %(interface)s: %(error)s', {'interface': interface, 'error': e})
    return LOCALHOST
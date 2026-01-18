import ipaddress
import socket
from openstack import _log
from openstack import exceptions
from openstack import utils
def _make_address_dict(fip, port):
    address = dict(version=4, addr=fip['floating_ip_address'])
    address['OS-EXT-IPS:type'] = 'floating'
    address['OS-EXT-IPS-MAC:mac_addr'] = port['mac_address']
    return address
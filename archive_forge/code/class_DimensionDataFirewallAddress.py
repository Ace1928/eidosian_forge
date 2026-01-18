from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataFirewallAddress:
    """
    The source or destination model in a firewall rule
    """

    def __init__(self, any_ip, ip_address, ip_prefix_size, port_begin, port_end, address_list_id, port_list_id):
        self.any_ip = any_ip
        self.ip_address = ip_address
        self.ip_prefix_size = ip_prefix_size
        self.port_list_id = port_list_id
        self.port_begin = port_begin
        self.port_end = port_end
        self.address_list_id = address_list_id
        self.port_list_id = port_list_id

    def __repr__(self):
        return '<DimensionDataFirewallAddress: any_ip=%s, ip_address=%s, ip_prefix_size=%s, port_begin=%s, port_end=%s, address_list_id=%s, port_list_id=%s>' % (self.any_ip, self.ip_address, self.ip_prefix_size, self.port_begin, self.port_end, self.address_list_id, self.port_list_id)
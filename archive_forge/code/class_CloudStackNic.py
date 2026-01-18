import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNic:
    """
    Class representing a CloudStack Network Interface.
    """

    def __init__(self, id, network_id, net_mask, gateway, ip_address, is_default, mac_address, driver, extra=None):
        self.id = id
        self.network_id = network_id
        self.net_mask = net_mask
        self.gateway = gateway
        self.ip_address = ip_address
        self.is_default = is_default
        self.mac_address = mac_address
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackNic: id=%s, network_id=%s, net_mask=%s, gateway=%s, ip_address=%s, is_default=%s, mac_address=%s, driver%s>' % (self.id, self.network_id, self.net_mask, self.gateway, self.ip_address, self.is_default, self.mac_address, self.driver.name)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id
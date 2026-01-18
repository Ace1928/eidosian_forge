import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNetworkOffering:
    """
    Class representing a CloudStack Network Offering.
    """

    def __init__(self, name, display_text, guest_ip_type, id, service_offering_id, for_vpc, driver, extra=None):
        self.display_text = display_text
        self.name = name
        self.guest_ip_type = guest_ip_type
        self.id = id
        self.service_offering_id = service_offering_id
        self.for_vpc = for_vpc
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackNetworkOffering: id=%s, name=%s, display_text=%s, guest_ip_type=%s, service_offering_id=%s, for_vpc=%s, driver=%s>' % (self.id, self.name, self.display_text, self.guest_ip_type, self.service_offering_id, self.for_vpc, self.driver.name)
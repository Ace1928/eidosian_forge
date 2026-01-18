import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNetwork:
    """
    Class representing a CloudStack Network.
    """

    def __init__(self, displaytext, name, networkofferingid, id, zoneid, driver, extra=None):
        self.displaytext = displaytext
        self.name = name
        self.networkofferingid = networkofferingid
        self.id = id
        self.zoneid = zoneid
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackNetwork: displaytext=%s, name=%s, networkofferingid=%s, id=%s, zoneid=%s, driver=%s>' % (self.displaytext, self.name, self.networkofferingid, self.id, self.zoneid, self.driver.name)
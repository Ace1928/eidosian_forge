import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackProject:
    """
    Class representing a CloudStack Project.
    """

    def __init__(self, id, name, display_text, driver, extra=None):
        self.id = id
        self.name = name
        self.display_text = display_text
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<CloudStackProject: id=%s, name=%s, display_text=%s,driver=%s>' % (self.id, self.display_text, self.name, self.driver.name)
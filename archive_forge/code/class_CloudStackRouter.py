import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackRouter:
    """
    Class representing a CloudStack Router.
    """

    def __init__(self, id, name, state, public_ip, vpc_id, driver):
        self.id = id
        self.name = name
        self.state = state
        self.public_ip = public_ip
        self.vpc_id = vpc_id
        self.driver = driver

    def __repr__(self):
        return '<CloudStackRouter: id=%s, name=%s, state=%s, public_ip=%s, vpc_id=%s, driver=%s>' % (self.id, self.name, self.state, self.public_ip, self.vpc_id, self.driver.name)
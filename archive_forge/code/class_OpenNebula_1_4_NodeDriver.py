import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebula_1_4_NodeDriver(OpenNebulaNodeDriver):
    """
    OpenNebula.org node driver for OpenNebula.org v1.4.
    """
    name = 'OpenNebula (v1.4)'
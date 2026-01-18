import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStackKeyPair:
    """
    A KeyPair.
    """

    def __init__(self, name, fingerprint, public_key, driver, private_key=None, extra=None):
        """
        Constructor.

        :keyword    name: Name of the KeyPair.
        :type       name: ``str``

        :keyword    fingerprint: Fingerprint of the KeyPair
        :type       fingerprint: ``str``

        :keyword    public_key: Public key in OpenSSH format.
        :type       public_key: ``str``

        :keyword    private_key: Private key in PEM format.
        :type       private_key: ``str``

        :keyword    extra: Extra attributes associated with this KeyPair.
        :type       extra: ``dict``
        """
        self.name = name
        self.fingerprint = fingerprint
        self.public_key = public_key
        self.private_key = private_key
        self.driver = driver
        self.extra = extra or {}

    def __repr__(self):
        return '<OpenStackKeyPair name={} fingerprint={} public_key={} ...>'.format(self.name, self.fingerprint, self.public_key)
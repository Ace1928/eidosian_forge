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
class OpenStack_2_QuotaSetItem:
    """
    Qouta Set Item info. Each item has three attributes: in_use,
    limit and reserved.

    See:
    https://docs.openstack.org/api-ref/compute/?expanded=show-the-detail-of-quota-detail#show-a-quota
    """

    def __init__(self, in_use, limit, reserved):
        """
        :param in_use: Number of currently used resources.
        :type in_use: ``int``
        :param limit: Max number of available resources.
        :type limit: ``int``
        :param reserved: Number of reserved resources.
        :type reserved: ``int``
        """
        self.in_use = in_use
        self.limit = limit
        self.reserved = reserved

    def __repr__(self):
        return '<OpenStack_2_QuotaSetItem in_use="%s", limit="%s",reserved="%s">' % (self.in_use, self.limit, self.reserved)
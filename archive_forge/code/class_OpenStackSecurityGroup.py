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
class OpenStackSecurityGroup:
    """
    A Security Group.
    """

    def __init__(self, id, tenant_id, name, description, driver, rules=None, extra=None):
        """
        Constructor.

        :keyword    id: Group id.
        :type       id: ``str``

        :keyword    tenant_id: Owner of the security group.
        :type       tenant_id: ``str``

        :keyword    name: Human-readable name for the security group. Might
                          not be unique.
        :type       name: ``str``

        :keyword    description: Human-readable description of a security
                                 group.
        :type       description: ``str``

        :keyword    rules: Rules associated with this group.
        :type       rules: ``list`` of
                    :class:`OpenStackSecurityGroupRule`

        :keyword    extra: Extra attributes associated with this group.
        :type       extra: ``dict``
        """
        self.id = id
        self.tenant_id = tenant_id
        self.name = name
        self.description = description
        self.driver = driver
        self.rules = rules or []
        self.extra = extra or {}

    def __repr__(self):
        return '<OpenStackSecurityGroup id=%s tenant_id=%s name=%s         description=%s>' % (self.id, self.tenant_id, self.name, self.description)
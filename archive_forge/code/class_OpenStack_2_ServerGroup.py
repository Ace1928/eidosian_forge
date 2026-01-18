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
class OpenStack_2_ServerGroup:
    """
    Server Group info.

    See:
    https://docs.openstack.org/api-ref/compute/?expanded=create-server-detail,list-server-groups-detail#server-groups-os-server-groups
    """

    def __init__(self, id, name, policy, members=None, rules=None, driver=None):
        """
        :param id: Server Group ID.
        :type id: ``str``
        :param name: Server Group Name.
        :type name: ``str``
        :param policy: Server Group policy.
        :type policy: ``str``
        :param members: Server Group members.
        :type members: ``list``
        :param rules: Server Group rules.
        :type rules: ``list``
        """
        self.id = id
        self.name = name
        self.policy = policy
        self.members = members or []
        self.rules = rules or []
        self.driver = driver

    def __repr__(self):
        return '<OpenStack_2_ServerGroup id="%s", name="%s", policy="%s", members="%s", rules="%s">' % (self.id, self.name, self.policy, self.members, self.rules)
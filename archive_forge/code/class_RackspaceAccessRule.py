from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
class RackspaceAccessRule:
    """
    An access rule allows or denies traffic to a Load Balancer based on the
    incoming IPs.

    :param id: Unique identifier to refer to this rule by.
    :type id: ``str``

    :param rule_type: RackspaceAccessRuleType.ALLOW or
                      RackspaceAccessRuleType.DENY.
    :type id: ``int``

    :param address: IP address or cidr (can be IPv4 or IPv6).
    :type address: ``str``
    """

    def __init__(self, id=None, rule_type=None, address=None):
        self.id = id
        self.rule_type = rule_type
        self.address = address

    def _to_dict(self):
        type_string = RackspaceAccessRuleType._RULE_TYPE_STRING_MAP[self.rule_type]
        as_dict = {'type': type_string, 'address': self.address}
        if self.id is not None:
            as_dict['id'] = self.id
        return as_dict
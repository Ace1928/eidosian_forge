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
class OpenStack_2_NetworkQuota:
    """
    Network Quota info. To get the information about quotas and used resources.

    See:
    https://docs.openstack.org/api-ref/network/v2/?expanded=show-quota-details-for-a-tenant-detail,list-quotas-for-a-project-detail#show-quota-details-for-a-tenant

    """

    def __init__(self, floatingip, network, port, rbac_policy, router, security_group, security_group_rule, subnet, subnetpool, driver=None):
        """
        :param floatingip: Quota of floating ips.
        :type floatingip: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param network: Quota of networks.
        :type network: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param port: Quota of ports.
        :type port: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param rbac_policy: Quota of rbac policies.
        :type rbac_policy: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param router: Quota of routers.
        :type router: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param security_group: Quota of security groups.
        :type security_group: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param security_group_rule: Quota of security group rules.
        :type security_group_rule: :class:`.OpenStack_2_QuotaSetItem`
                                   or ``dict``
        :param subnet: Quota of subnets.
        :type subnet: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        :param subnetpool: Quota of subnet pools.
        :type subnetpool: :class:`.OpenStack_2_QuotaSetItem` or ``dict``
        """
        self.floatingip = self._to_quota_set_item(floatingip)
        self.network = self._to_quota_set_item(network)
        self.port = self._to_quota_set_item(port)
        self.rbac_policy = self._to_quota_set_item(rbac_policy)
        self.router = self._to_quota_set_item(router)
        self.security_group = self._to_quota_set_item(security_group)
        self.security_group_rule = self._to_quota_set_item(security_group_rule)
        self.subnet = self._to_quota_set_item(subnet)
        self.subnetpool = self._to_quota_set_item(subnetpool)
        self.driver = driver

    def _to_quota_set_item(self, obj):
        if obj:
            if isinstance(obj, OpenStack_2_QuotaSetItem):
                return obj
            elif isinstance(obj, dict):
                return OpenStack_2_QuotaSetItem(obj['used'], obj['limit'], obj['reserved'])
        else:
            return None

    def __repr__(self):
        return '<OpenStack_2_NetworkQuota Floating IPs="%s", networks="%s", SGs="%s", SGRs="%s">' % (self.floatingip, self.network, self.security_group, self.security_group_rule)
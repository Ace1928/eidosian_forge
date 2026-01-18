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
def _kwargs_to_mutable_member_attrs(self, **attrs):
    update_attrs = {}
    if 'condition' in attrs:
        update_attrs['condition'] = self.CONDITION_LB_MEMBER_MAP.get(attrs['condition'])
    if 'weight' in attrs:
        update_attrs['weight'] = attrs['weight']
    return update_attrs
from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_member(self, obj, port, balancer):
    return Member(id=obj['id'], ip=obj['nic'][0]['ipaddress'], port=port, balancer=balancer)
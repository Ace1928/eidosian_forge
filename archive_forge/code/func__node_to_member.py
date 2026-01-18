from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.brightbox import BrightboxConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _node_to_member(self, data, balancer):
    return Member(id=data['id'], ip=None, port=None, balancer=balancer)
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.brightbox import BrightboxConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _member_to_node(self, member):
    return {'node': member.id}
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.brightbox import BrightboxConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _public_ip(self, data):
    if len(data['cloud_ips']) > 0:
        ip = data['cloud_ips'][0]['public_ip']
    else:
        ip = None
    return ip
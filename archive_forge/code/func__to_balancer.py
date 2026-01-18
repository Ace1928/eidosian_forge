from libcloud.utils.misc import reverse_dict
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def _to_balancer(self, obj):
    balancer = LoadBalancer(id=obj['id'], name=obj['name'], state=self.LB_STATE_MAP.get(obj['state'], State.UNKNOWN), ip=obj['publicip'], port=obj['publicport'], driver=self.connection.driver)
    balancer.ex_private_port = obj['privateport']
    balancer.ex_public_ip_id = obj['publicipid']
    return balancer
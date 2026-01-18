from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_list_balancer_tags(self, balancer_id):
    params = {'Action': 'DescribeTags', 'LoadBalancerNames.member.1': balancer_id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_tags(data)
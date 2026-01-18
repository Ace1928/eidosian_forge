from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_rules(self, data):
    xpath = 'DescribeRulesResult/Rules/member'
    return [self._to_rule(el) for el in findall(element=data, xpath=xpath, namespace=NS)]
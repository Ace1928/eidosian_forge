from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_policy_types(self, data):
    xpath = 'DescribeLoadBalancerPolicyTypesResult/'
    xpath += 'PolicyTypeDescriptions/member'
    return [findtext(element=el, xpath='PolicyTypeName', namespace=NS) for el in findall(element=data, xpath=xpath, namespace=NS)]
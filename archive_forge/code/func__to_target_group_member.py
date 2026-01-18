from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_target_group_member(self, el):
    member = Member(id=findtext(element=el, xpath='Target/Id', namespace=NS), ip=None, port=findtext(element=el, xpath='Target/Port', namespace=NS), balancer=None, extra={'health': findtext(element=el, xpath='TargetHealth/State', namespace=NS)})
    return member
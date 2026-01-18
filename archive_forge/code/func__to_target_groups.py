from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_target_groups(self, data):
    xpath = 'DescribeTargetGroupsResult/TargetGroups/member'
    return [self._to_target_group(el) for el in findall(element=data, xpath=xpath, namespace=NS)]
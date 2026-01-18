from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_target_group(self, el):
    target_group = ALBTargetGroup(target_group_id=findtext(element=el, xpath='TargetGroupArn', namespace=NS), name=findtext(element=el, xpath='TargetGroupName', namespace=NS), protocol=findtext(element=el, xpath='Protocol', namespace=NS), port=int(findtext(element=el, xpath='Port', namespace=NS)), vpc=findtext(element=el, xpath='VpcId', namespace=NS), driver=self.connection.driver, health_check_timeout=int(findtext(element=el, xpath='HealthCheckTimeoutSeconds', namespace=NS)), health_check_port=findtext(element=el, xpath='HealthCheckPort', namespace=NS), health_check_path=findtext(element=el, xpath='HealthCheckPath', namespace=NS), health_check_proto=findtext(element=el, xpath='HealthCheckProtocol', namespace=NS), health_check_interval=int(findtext(element=el, xpath='HealthCheckIntervalSeconds', namespace=NS)), healthy_threshold=int(findtext(element=el, xpath='HealthyThresholdCount', namespace=NS)), unhealthy_threshold=int(findtext(element=el, xpath='UnhealthyThresholdCount', namespace=NS)), health_check_matcher=findtext(element=el, xpath='Matcher/HttpCode', namespace=NS))
    lbs = findall(element=el, xpath='LoadBalancerArns/member', namespace=NS)
    target_group._balancers_arns = [lb_arn.text for lb_arn in lbs]
    return target_group
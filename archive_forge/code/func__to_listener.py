from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_listener(self, el):
    listener = ALBListener(listener_id=findtext(element=el, xpath='ListenerArn', namespace=NS), protocol=findtext(element=el, xpath='Protocol', namespace=NS), port=int(findtext(element=el, xpath='Port', namespace=NS)), balancer=None, driver=self.connection.driver, action=findtext(element=el, xpath='DefaultActions/member/Type', namespace=NS), ssl_policy=findtext(element=el, xpath='SslPolicy', namespace=NS), ssl_certificate=findtext(element=el, xpath='Certificates/member/CertificateArn', namespace=NS))
    listener._balancer_arn = findtext(element=el, xpath='LoadBalancerArn', namespace=NS)
    return listener
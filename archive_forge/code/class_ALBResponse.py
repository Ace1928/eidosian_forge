from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ALBResponse(AWSGenericResponse):
    """
    Amazon ALB response class.
    """
    namespace = NS
    exceptions = {}
    xpath = 'Error'
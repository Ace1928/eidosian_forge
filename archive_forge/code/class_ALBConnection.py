from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ALBConnection(SignedAWSConnection):
    version = VERSION
    host = HOST
    responseCls = ALBResponse
    service_name = 'elasticloadbalancing'
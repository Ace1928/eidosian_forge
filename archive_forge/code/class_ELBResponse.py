from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ELBResponse(AWSGenericResponse):
    """
    Amazon ELB response class.
    """
    namespace = NS
    exceptions = {}
    xpath = 'Error'
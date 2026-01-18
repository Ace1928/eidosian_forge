from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBLoadBalancerTcpListener(SLBLoadBalancerListener):
    """
    This class represents a rule to route tcp request to the backends.
    """
    action = 'CreateLoadBalancerTCPListener'
    option_keys = ['PersistenceTimeout', 'HealthCheckType', 'HealthCheckDomain', 'HealthCheckURI', 'HealthCheckConnectPort', 'HealthyThreshold', 'UnhealthyThreshold', 'HealthCheckConnectTimeout', 'HealthCheckInterval', 'HealthCheckHttpCode']
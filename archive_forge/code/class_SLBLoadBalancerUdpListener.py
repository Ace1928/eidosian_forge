from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBLoadBalancerUdpListener(SLBLoadBalancerTcpListener):
    """
    This class represents a rule to route udp request to the backends.
    """
    action = 'CreateLoadBalancerUDPListener'
    option_keys = ['PersistenceTimeout', 'HealthCheckConnectPort', 'HealthyThreshold', 'UnhealthyThreshold', 'HealthCheckConnectTimeout', 'HealthCheckInterval']
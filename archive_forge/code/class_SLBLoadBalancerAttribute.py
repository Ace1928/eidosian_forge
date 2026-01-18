from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBLoadBalancerAttribute:
    """
    This class used to get listeners and backend servers related to a balancer
    listeners is a ``list`` of ``dict``, each element contains
    'ListenerPort' and 'ListenerProtocol' keys.
    backend_servers is a ``list`` of ``dict``, each element contains
    'ServerId' and 'Weight' keys.
    """

    def __init__(self, balancer, listeners, backend_servers, extra=None):
        self.balancer = balancer
        self.listeners = listeners or []
        self.backend_servers = backend_servers or []
        self.extra = extra or {}

    def is_listening(self, port):
        for listener in self.listeners:
            if listener.get('ListenerPort') == port:
                return True
        return False

    def is_attached(self, member):
        for server in self.backend_servers:
            if server.get('Serverid') == member.id:
                return True
        return False

    def __repr__(self):
        return '<SLBLoadBalancerAttribute id={}, ports={}, servers={} ...>'.format(self.balancer.id, self.listeners, self.backend_servers)
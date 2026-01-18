from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_port_and_protocol(self, el):
    port = findtext(el, 'ListenerPort', namespace=self.namespace)
    protocol = findtext(el, 'ListenerProtocol', namespace=self.namespace)
    return {'ListenerPort': port, 'ListenerProtocol': protocol}
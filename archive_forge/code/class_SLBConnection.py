from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBConnection(SignedAliyunConnection):
    api_version = SLB_API_VERSION
    host = SLB_API_HOST
    responseCls = AliyunXmlResponse
    service_name = 'slb'
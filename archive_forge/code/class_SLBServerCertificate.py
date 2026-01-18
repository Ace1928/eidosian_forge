from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBServerCertificate(ReprMixin):
    _repr_attributes = ['id', 'name', 'fingerprint']

    def __init__(self, id, name, fingerprint):
        self.id = id
        self.name = name
        self.fingerprint = fingerprint
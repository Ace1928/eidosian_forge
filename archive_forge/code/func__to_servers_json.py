from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_servers_json(self, members):
    servers = []
    for each in members:
        server = {'ServerId': each.id, 'Weight': '100'}
        if 'Weight' in each.extra:
            server['Weight'] = each.extra['Weight']
        servers.append(server)
    try:
        return json.dumps(servers)
    except Exception:
        raise AttributeError('could not convert member to backend server')
from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class NameServerController(V2Controller):

    def list(self, zone):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        url = f'/zones/{zone}/nameservers'
        return self._get(url, response_key='nameservers')
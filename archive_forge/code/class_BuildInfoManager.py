from heatclient.common import base
from heatclient.common import utils
class BuildInfoManager(base.BaseManager):
    resource_class = BuildInfo

    def build_info(self):
        resp = self.client.get('/build_info')
        body = utils.get_response_body(resp)
        return body
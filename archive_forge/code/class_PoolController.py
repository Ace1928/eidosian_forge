from designateclient.v2.base import V2Controller
class PoolController(V2Controller):

    def list(self):
        url = '/pools'
        return self._get(url, response_key='pools')
from designateclient.v2.base import V2Controller
class LimitController(V2Controller):

    def get(self):
        return self._get('/limits')
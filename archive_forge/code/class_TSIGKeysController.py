from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class TSIGKeysController(V2Controller):

    def create(self, name, algorithm, secret, scope, resource_id):
        data = {'name': name, 'algorithm': algorithm, 'secret': secret, 'scope': scope, 'resource_id': resource_id}
        return self._post('/tsigkeys', data=data)

    def list(self, criterion=None, marker=None, limit=None):
        url = self.build_url('/tsigkeys', criterion, marker, limit)
        return self._get(url, response_key='tsigkeys')

    def get(self, tsigkey):
        tsigkey = v2_utils.resolve_by_name(self.list, tsigkey)
        return self._get(f'/tsigkeys/{tsigkey}')

    def update(self, tsigkey, values):
        tsigkey = v2_utils.resolve_by_name(self.list, tsigkey)
        return self._patch(f'/tsigkeys/{tsigkey}', data=values)

    def delete(self, tsigkey):
        tsigkey = v2_utils.resolve_by_name(self.list, tsigkey)
        return self._delete(f'/tsigkeys/{tsigkey}')
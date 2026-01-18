from designateclient.v2.base import V2Controller
class BlacklistController(V2Controller):

    def create(self, pattern, description=None):
        data = {'pattern': pattern}
        if description is not None:
            data['description'] = description
        return self._post('/blacklists', data=data)

    def list(self, criterion=None, marker=None, limit=None):
        url = self.build_url('/blacklists', criterion, marker, limit)
        return self._get(url, response_key='blacklists')

    def get(self, blacklist_id):
        url = f'/blacklists/{blacklist_id}'
        return self._get(url)

    def update(self, blacklist_id, values):
        url = f'/blacklists/{blacklist_id}'
        return self._patch(url, data=values)

    def delete(self, blacklist_id):
        url = f'/blacklists/{blacklist_id}'
        return self._delete(url)
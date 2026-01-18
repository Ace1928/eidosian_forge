from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class ZoneController(V2Controller):

    def create(self, name, type_=None, email=None, description=None, ttl=None, masters=None, attributes=None):
        type_ = type_ or 'PRIMARY'
        data = {'name': name, 'type': type_}
        if type_ == 'PRIMARY':
            if email:
                data['email'] = email
            if ttl is not None:
                data['ttl'] = ttl
        elif type_ == 'SECONDARY' and masters:
            data['masters'] = masters
        if description is not None:
            data['description'] = description
        if attributes is not None:
            data['attributes'] = attributes
        return self._post('/zones', data=data)

    def list(self, criterion=None, marker=None, limit=None):
        url = self.build_url('/zones', criterion, marker, limit)
        return self._get(url, response_key='zones')

    def get(self, zone):
        zone = v2_utils.resolve_by_name(self.list, zone)
        return self._get(f'/zones/{zone}')

    def update(self, zone, values):
        zone = v2_utils.resolve_by_name(self.list, zone)
        url = self.build_url(f'/zones/{zone}')
        return self._patch(url, data=values)

    def delete(self, zone, delete_shares=False):
        zone = v2_utils.resolve_by_name(self.list, zone)
        url = self.build_url(f'/zones/{zone}')
        if delete_shares:
            headers = {'X-Designate-Delete-Shares': 'true'}
            _resp, body = self.client.session.delete(url, headers=headers)
        else:
            _resp, body = self.client.session.delete(url)
        return body

    def abandon(self, zone):
        zone = v2_utils.resolve_by_name(self.list, zone)
        url = f'/zones/{zone}/tasks/abandon'
        self.client.session.post(url)

    def axfr(self, zone):
        zone = v2_utils.resolve_by_name(self.list, zone)
        url = f'/zones/{zone}/tasks/xfr'
        self.client.session.post(url)

    def pool_move(self, zone, values):
        zone = v2_utils.resolve_by_name(self.list, zone)
        url = self.build_url('/zones/%s/tasks/pool_move' % zone)
        return self._post(url, data=values)
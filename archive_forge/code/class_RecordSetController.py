from oslo_utils import uuidutils
from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class RecordSetController(V2Controller):

    def _canonicalize_record_name(self, zone, name):
        zone_info = None
        if isinstance(zone, str) and (not uuidutils.is_uuid_like(zone)):
            zone_info = self.client.zones.get(zone)
        elif isinstance(zone, dict):
            zone_info = zone
        if not name.endswith('.'):
            if not isinstance(zone_info, dict):
                zone_info = self.client.zones.get(zone)
            name = '{}.{}'.format(name, zone_info['name'])
        return (name, zone_info)

    def create(self, zone, name, type_, records, description=None, ttl=None):
        name, zone_info = self._canonicalize_record_name(zone, name)
        data = {'name': name, 'type': type_, 'records': records}
        if ttl is not None:
            data['ttl'] = ttl
        if description is not None:
            data['description'] = description
        if zone_info is not None:
            zone_id = zone_info['id']
        else:
            zone_id = zone
        return self._post(f'/zones/{zone_id}/recordsets', data=data)

    def list(self, zone, criterion=None, marker=None, limit=None):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        url = self.build_url(f'/zones/{zone}/recordsets', criterion, marker, limit)
        return self._get(url, response_key='recordsets')

    def list_all_zones(self, criterion=None, marker=None, limit=None):
        url = self.build_url('/recordsets', criterion, marker, limit)
        return self._get(url, response_key='recordsets')

    def get(self, zone, recordset):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        recordset = v2_utils.resolve_by_name(self.list, recordset, zone)
        url = self.build_url(f'/zones/{zone}/recordsets/{recordset}')
        return self._get(url)

    def update(self, zone, recordset, values):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        recordset = v2_utils.resolve_by_name(self.list, recordset, zone)
        url = f'/zones/{zone}/recordsets/{recordset}'
        return self._put(url, data=values)

    def delete(self, zone, recordset):
        zone = v2_utils.resolve_by_name(self.client.zones.list, zone)
        recordset = v2_utils.resolve_by_name(self.list, recordset, zone)
        url = f'/zones/{zone}/recordsets/{recordset}'
        return self._delete(url)
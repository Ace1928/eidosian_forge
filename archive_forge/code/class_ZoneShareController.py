from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
class ZoneShareController(V2Controller):

    def create(self, zone, target_project_id):
        zone_id = v2_utils.resolve_by_name(self.client.zones.list, zone)
        data = {'target_project_id': target_project_id}
        return self._post(f'/zones/{zone_id}/shares', data=data)

    def list(self, zone, criterion=None, marker=None, limit=None):
        zone_id = v2_utils.resolve_by_name(self.client.zones.list, zone)
        url = self.build_url(f'/zones/{zone_id}/shares', criterion, marker, limit)
        return self._get(url, response_key='shared_zones')

    def delete(self, zone, shared_zone_id):
        zone_id = v2_utils.resolve_by_name(self.client.zones.list, zone)
        return self._delete(f'/zones/{zone_id}/shares/{shared_zone_id}')

    def get(self, zone, shared_zone_id):
        zone_id = v2_utils.resolve_by_name(self.client.zones.list, zone)
        return self._get(f'/zones/{zone_id}/shares/{shared_zone_id}')
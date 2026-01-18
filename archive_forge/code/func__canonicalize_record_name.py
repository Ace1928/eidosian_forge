from oslo_utils import uuidutils
from designateclient.v2.base import V2Controller
from designateclient.v2 import utils as v2_utils
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
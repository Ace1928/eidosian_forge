import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def _normalize_server(self, server):
    ret = utils.Munch()
    server = utils.Munch(server)
    self._remove_novaclient_artifacts(server)
    ret['id'] = server.pop('id')
    ret['name'] = server.pop('name')
    server['flavor'].pop('links', None)
    ret['flavor'] = server.pop('flavor')
    server.pop('flavorRef', None)
    image = server.pop('image', None)
    if str(image) != image:
        image = utils.Munch(id=image['id'])
    ret['image'] = image
    server.pop('imageRef', None)
    ret['block_device_mapping'] = server.pop('block_device_mapping_v2', {})
    project_id = server.pop('tenant_id', '')
    project_id = server.pop('project_id', project_id)
    az = _pop_or_get(server, 'OS-EXT-AZ:availability_zone', None, self.strict_mode)
    ret['location'] = server.pop('location', self._get_current_location(project_id=project_id, zone=az))
    ret['volumes'] = _pop_or_get(server, 'os-extended-volumes:volumes_attached', [], self.strict_mode)
    config_drive = server.pop('has_config_drive', server.pop('config_drive', False))
    ret['has_config_drive'] = _to_bool(config_drive)
    host_id = server.pop('hostId', server.pop('host_id', None))
    ret['host_id'] = host_id
    ret['progress'] = _pop_int(server, 'progress')
    ret['disk_config'] = _pop_or_get(server, 'OS-DCF:diskConfig', None, self.strict_mode)
    for key in ('OS-EXT-STS:power_state', 'OS-EXT-STS:task_state', 'OS-EXT-STS:vm_state', 'OS-SRV-USG:launched_at', 'OS-SRV-USG:terminated_at', 'OS-EXT-SRV-ATTR:hypervisor_hostname', 'OS-EXT-SRV-ATTR:instance_name', 'OS-EXT-SRV-ATTR:user_data', 'OS-EXT-SRV-ATTR:host', 'OS-EXT-SRV-ATTR:hostname', 'OS-EXT-SRV-ATTR:kernel_id', 'OS-EXT-SRV-ATTR:launch_index', 'OS-EXT-SRV-ATTR:ramdisk_id', 'OS-EXT-SRV-ATTR:reservation_id', 'OS-EXT-SRV-ATTR:root_device_name', 'OS-SCH-HNT:scheduler_hints'):
        short_key = key.split(':')[1]
        ret[short_key] = _pop_or_get(server, key, None, self.strict_mode)
    ret['security_groups'] = server.pop('security_groups', None) or []
    ret['created_at'] = server.get('created')
    for field in _SERVER_FIELDS:
        ret[field] = server.pop(field, None)
    if not ret['networks']:
        ret['networks'] = {}
    ret['interface_ip'] = ''
    ret['properties'] = server.copy()
    if not self.strict_mode:
        ret['hostId'] = host_id
        ret['config_drive'] = config_drive
        ret['project_id'] = project_id
        ret['tenant_id'] = project_id
        ret['region'] = self.config.get_region_name('compute')
        ret['cloud'] = self.config.name
        ret['az'] = az
        for key, val in ret['properties'].items():
            ret.setdefault(key, val)
    return ret
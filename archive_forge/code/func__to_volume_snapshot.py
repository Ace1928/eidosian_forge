import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def _to_volume_snapshot(self, data):
    extra = {'created_at': data['created_at'], 'resource_id': data['resource_id'], 'regions': data['regions'], 'min_disk_size': data['min_disk_size']}
    return VolumeSnapshot(id=data['id'], name=data['name'], size=data['size_gigabytes'], driver=self, extra=extra)
import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_volume_quota(self, obj):
    res = OpenStack_2_VolumeQuota(backup_gigabytes=obj.get('backup_gigabytes', None), gigabytes=obj.get('gigabytes', None), per_volume_gigabytes=obj.get('per_volume_gigabytes', None), backups=obj.get('backups', None), snapshots=obj.get('snapshots', None), volumes=obj.get('volumes', None), driver=self.connection.driver)
    return res
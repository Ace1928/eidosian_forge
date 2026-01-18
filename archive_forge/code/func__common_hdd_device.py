import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
def _common_hdd_device(self):
    return {'title': self.image.name, 'size': self.size.disk, 'tier': self.size.extra.get('storage_tier', 'maxiops')}
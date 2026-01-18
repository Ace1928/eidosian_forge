from typing import Any, Dict, Optional
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import VolumeSnapshot
class VultrNodeSnapshot(VolumeSnapshot):

    def __repr__(self):
        return '<VultrNodeSnapshot id={} size={} driver={} state={}>'.format(self.id, self.size, self.driver.name, self.state)
import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def _to_volume_state(self, vol):
    state = self.VOLUME_STATE_MAP.get(vol['state'], StorageVolumeState.UNKNOWN)
    if state == StorageVolumeState.AVAILABLE and 'virtualmachineid' in vol:
        state = StorageVolumeState.INUSE
    return state
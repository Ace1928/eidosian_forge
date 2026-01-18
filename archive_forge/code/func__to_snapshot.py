import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _to_snapshot(self, snapshot):
    state = self.SNAPSHOT_STATE_MAP.get(snapshot['state'], VolumeSnapshotState.UNKNOWN)
    extra = {'organization': snapshot['organization'], 'volume_type': snapshot['volume_type']}
    return VolumeSnapshot(id=snapshot['id'], driver=self, size=_to_lib_size(snapshot['size']), created=parse_date(snapshot['creation_date']), state=state, extra=extra)
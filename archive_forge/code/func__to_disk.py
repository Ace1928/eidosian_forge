from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _to_disk(self, element):
    disk = Disk(id=element['id'], state=NODE_STATE_MAP.get(element['state'], NodeState.UNKNOWN), name=element['name'], driver=self.connection.driver, size=element['size'], extra={'can_snapshot': element['can_snapshot']})
    return disk
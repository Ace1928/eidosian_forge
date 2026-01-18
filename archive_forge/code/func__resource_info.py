from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _resource_info(self, type, id):
    try:
        obj = self.connection.request('hosting.%s.info' % type, int(id))
        return obj.object
    except Exception as e:
        raise GandiException(1003, e)
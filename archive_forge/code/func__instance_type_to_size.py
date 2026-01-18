from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _instance_type_to_size(self, instance):
    return NodeSize(id=instance['id'], name=instance['name'], ram=instance['memory'], disk=instance['disk'], bandwidth=instance['bandwidth'], price=self._get_size_price(size_id=instance['id']), driver=self.connection.driver)
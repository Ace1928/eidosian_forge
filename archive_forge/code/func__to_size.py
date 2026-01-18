import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_size(self, size):
    sizes = []
    for disk in size['disks']:
        sizes.append(NodeSize(id=str(size['id']), name=size['name'], ram=size['memory'], disk=disk, driver=self, extra={'vcpus': size['vcpus']}, bandwidth=0, price=0))
    return sizes
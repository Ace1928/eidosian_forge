import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_network(self, network):
    return G8Network(str(network['id']), network['name'], None, network['externalnetworkip'], self)
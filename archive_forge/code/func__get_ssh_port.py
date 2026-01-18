import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _get_ssh_port(forwards, node):
    for forward in forwards:
        if forward.node_id == node.id and forward.privateport == 22:
            return forward
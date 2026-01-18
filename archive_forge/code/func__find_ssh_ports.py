import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _find_ssh_ports(self, ex_network, node):
    forwards = ex_network.list_portforwards()
    usedports = []
    result = {'node': None, 'network': usedports}
    for forward in forwards:
        usedports.append(forward.publicport)
        if forward.node_id == node.id and forward.privateport == 22:
            result['node'] = forward.privateport
    return result
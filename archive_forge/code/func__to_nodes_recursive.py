import ssl
import json
import time
import atexit
import base64
import asyncio
import hashlib
import logging
import warnings
import functools
import itertools
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_nodes_recursive(self, vm_list):
    nodes = []
    for virtual_machine in vm_list:
        if hasattr(virtual_machine, 'childEntity'):
            nodes.extend(self._to_nodes_recursive(virtual_machine.childEntity))
        elif isinstance(virtual_machine, vim.VirtualApp):
            nodes.extend(self._to_nodes_recursive(virtual_machine.vm))
        else:
            if not hasattr(virtual_machine, 'config') or (virtual_machine.config and virtual_machine.config.template):
                continue
            nodes.append(self._to_node_recursive(virtual_machine))
    return nodes
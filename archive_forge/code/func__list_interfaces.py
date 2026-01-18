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
def _list_interfaces(self):
    request = '/rest/appliance/networking/interfaces'
    response = self._request(request).object['value']
    interfaces = [{'name': interface['name'], 'mac': interface['mac'], 'status': interface['status'], 'ip': interface['ipv4']['address']} for interface in response]
    return interfaces
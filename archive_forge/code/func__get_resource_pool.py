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
def _get_resource_pool(self, host_id=None, cluster_id=None, name=None):
    if host_id:
        pms = {'filter.hosts': host_id}
    if cluster_id:
        pms = {'filter.clusters': cluster_id}
    if name:
        pms = {'filter.names': name}
    rp_request = '/rest/vcenter/resource-pool'
    resource_pool = self._request(rp_request, params=pms).object
    return resource_pool['value'][0]['resource_pool']
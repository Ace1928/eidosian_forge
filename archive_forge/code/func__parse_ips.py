import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
def _parse_ips(self, data):
    public_ips = []
    private_ips = []
    for address in data:
        if 'address' in address and address['address'] is not None:
            if 'public' in address and address['public'] is True:
                public_ips.append(address['address'])
            else:
                private_ips.append(address['address'])
    return {'public': public_ips, 'private': private_ips}
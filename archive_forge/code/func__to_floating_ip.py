import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def _to_floating_ip(self, obj):
    return DigitalOcean_v2_FloatingIpAddress(id=obj['ip'], ip_address=obj['ip'], node_id=obj['droplet']['id'] if obj['droplet'] else None, extra={'region': obj['region']}, driver=self)
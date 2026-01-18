import json
import time
import base64
from typing import Any, Dict, List, Union, Optional
from functools import update_wrapper
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError, ServiceUnavailableError
from libcloud.common.vultr import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.utils.publickey import get_pubkey_openssh_fingerprint
class VultrNodeDriver(NodeDriver):
    type = Provider.VULTR
    name = 'Vultr'
    website = 'https://www.vultr.com'

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, region=None, **kwargs):
        if cls is VultrNodeDriver:
            if api_version == '1':
                cls = VultrNodeDriverV1
            elif api_version == '2':
                cls = VultrNodeDriverV2
            else:
                raise NotImplementedError('No Vultr driver found for API version: %s' % api_version)
        return super().__new__(cls)
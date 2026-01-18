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
class VultrResponse(JsonResponse):

    def parse_error(self):
        if self.status == httplib.OK:
            body = self.parse_body()
            return body
        elif self.status == httplib.FORBIDDEN:
            raise InvalidCredsError(self.body)
        elif self.status == httplib.SERVICE_UNAVAILABLE:
            raise ServiceUnavailableError(self.body)
        else:
            raise LibcloudError(self.body)
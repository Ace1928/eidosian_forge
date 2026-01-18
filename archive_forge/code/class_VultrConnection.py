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
class VultrConnection(ConnectionKey):
    """
    Connection class for the Vultr driver.
    """
    host = 'api.vultr.com'
    responseCls = VultrResponse
    unauthenticated_endpoints = {'/v1/app/list': ['GET'], '/v1/os/list': ['GET'], '/v1/plans/list': ['GET'], '/v1/plans/list_vc2': ['GET'], '/v1/plans/list_vdc2': ['GET'], '/v1/regions/availability': ['GET'], '/v1/regions/list': ['GET']}

    def add_default_headers(self, headers):
        """
        Adds ``API-Key`` default header.

        :return: Updated headers.
        :rtype: dict
        """
        if self.require_api_key():
            headers.update({'API-Key': self.key})
        return headers

    def encode_data(self, data):
        return urlencode(data)

    @rate_limited()
    def get(self, url):
        return self.request(url)

    @rate_limited()
    def post(self, url, data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return self.request(url, data=data, headers=headers, method='POST')

    def require_api_key(self):
        """
        Check whether this call (method + action) must be authenticated.

        :return: True if ``API-Key`` header required, False otherwise.
        :rtype: bool
        """
        try:
            return self.method not in self.unauthenticated_endpoints[self.action]
        except KeyError:
            return True
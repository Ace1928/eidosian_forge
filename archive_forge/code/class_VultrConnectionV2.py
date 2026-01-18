from typing import Any, Dict, Optional
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import VolumeSnapshot
class VultrConnectionV2(ConnectionKey):
    """
    A connection to the Vultr API v2
    """
    host = API_HOST
    responseCls = VultrResponseV2

    def add_default_headers(self, headers):
        headers['Authorization'] = 'Bearer %s' % self.key
        headers['Content-Type'] = 'application/json'
        return headers

    def add_default_params(self, params):
        params['per_page'] = 500
        return params
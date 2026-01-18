import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio
class EquinixMetalConnection(ConnectionKey):
    """
    Connection class for the Equinix Metal driver.
    """
    host = EQUINIXMETAL_ENDPOINT
    responseCls = EquinixMetalResponse

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request
        """
        headers['Content-Type'] = 'application/json'
        headers['X-Auth-Token'] = self.key
        headers['X-Consumer-Token'] = 'kcrhMn7hwG8Ceo2hAhGFa2qpxLBvVHxEjS9ue8iqmsNkeeB2iQgMq4dNc1893pYu'
        return headers
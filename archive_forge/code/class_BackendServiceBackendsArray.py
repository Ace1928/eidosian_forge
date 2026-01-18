from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceBackendsArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({'balancingMode': item.get('balancing_mode'), 'capacityScaler': item.get('capacity_scaler'), 'description': item.get('description'), 'group': item.get('group'), 'maxConnections': item.get('max_connections'), 'maxConnectionsPerInstance': item.get('max_connections_per_instance'), 'maxConnectionsPerEndpoint': item.get('max_connections_per_endpoint'), 'maxRate': item.get('max_rate'), 'maxRatePerInstance': item.get('max_rate_per_instance'), 'maxRatePerEndpoint': item.get('max_rate_per_endpoint'), 'maxUtilization': item.get('max_utilization')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({'balancingMode': item.get('balancingMode'), 'capacityScaler': item.get('capacityScaler'), 'description': item.get('description'), 'group': item.get('group'), 'maxConnections': item.get('maxConnections'), 'maxConnectionsPerInstance': item.get('maxConnectionsPerInstance'), 'maxConnectionsPerEndpoint': item.get('maxConnectionsPerEndpoint'), 'maxRate': item.get('maxRate'), 'maxRatePerInstance': item.get('maxRatePerInstance'), 'maxRatePerEndpoint': item.get('maxRatePerEndpoint'), 'maxUtilization': item.get('maxUtilization')})
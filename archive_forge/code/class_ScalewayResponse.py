import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class ScalewayResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_error(self):
        return super().parse_error()['message']

    def success(self):
        return self.status in self.valid_response_codes
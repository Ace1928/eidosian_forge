import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.compute.types import Provider, NodeState
class CloudscaleResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_error(self):
        body = self.parse_body()
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(body['detail'])
        else:
            return next(iter(body.values()))

    def success(self):
        return self.status in self.valid_response_codes
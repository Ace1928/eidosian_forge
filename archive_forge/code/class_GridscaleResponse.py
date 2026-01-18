from libcloud.utils.py3 import httplib
from libcloud.common.base import BaseDriver, JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
class GridscaleResponse(JsonResponse):
    """
    Gridscale API Response
    """
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.NO_CONTENT]

    def parse_error(self):
        body = self.parse_body()
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError(body['message'])
        if self.status == httplib.NOT_FOUND:
            raise Exception('The resource you are looking for has not been found.')
        return self.body

    def success(self):
        return self.status in self.valid_response_codes
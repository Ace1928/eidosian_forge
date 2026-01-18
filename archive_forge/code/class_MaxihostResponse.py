from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
class MaxihostResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            body = self.parse_body()
            raise InvalidCredsError(body['message'])
        else:
            body = self.parse_body()
            if 'message' in body:
                error = '{} (code: {})'.format(body['message'], self.status)
            else:
                error = body
            return error

    def success(self):
        return self.status in self.valid_response_codes
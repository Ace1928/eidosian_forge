import json
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError
class PowerDNSResponse(JsonResponse):

    def success(self):
        i = int(self.status)
        return 200 <= i <= 299

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError('Invalid provider credentials')
        try:
            body = self.parse_body()
        except MalformedResponseError as e:
            body = '{}: {}'.format(e.value, e.body)
        try:
            errors = [body['error']]
        except TypeError:
            return '%s (HTTP Code: %d)' % (body, self.status)
        try:
            errors.append(body['errors'])
        except KeyError:
            pass
        return '%s (HTTP Code: %d)' % (' '.join(errors), self.status)
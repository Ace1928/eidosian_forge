from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyDNSException(LibcloudError):

    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.args = (code, message)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<GoDaddyDNSException in {}: {}>'.format(self.code, self.message)
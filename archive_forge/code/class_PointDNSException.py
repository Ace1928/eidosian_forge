from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
class PointDNSException(ProviderError):

    def __init__(self, value, http_code, driver=None):
        super().__init__(value=value, http_code=http_code, driver=driver)
        self.args = (http_code, value)
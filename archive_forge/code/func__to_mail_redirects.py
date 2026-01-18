from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def _to_mail_redirects(self, data, zone):
    mail_redirects = []
    for item in data:
        mail_redirect = self._to_mail_redirect(item, zone=zone)
        mail_redirects.append(mail_redirect)
    return mail_redirects
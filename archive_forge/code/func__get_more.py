import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.common.google import GoogleResponse, GoogleBaseConnection, ResourceNotFoundError
def _get_more(self, rtype, **kwargs):
    last_key = None
    exhausted = False
    while not exhausted:
        items, last_key, exhausted = self._get_data(rtype, last_key, **kwargs)
        yield from items
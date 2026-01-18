from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def _to_redirect(self, data, zone_id=None, zone=None):
    if not zone:
        zone = self.get_zone(zone_id)
    record = data.get('zone_redirect')
    id = record.get('id')
    name = record.get('name')
    redirect_to = record.get('redirect_to')
    type = record.get('redirect_type')
    iframe = record.get('iframe_title')
    query = record.get('redirect_query_string')
    return Redirect(id, name, redirect_to, type, self, zone, iframe=iframe, query=query)
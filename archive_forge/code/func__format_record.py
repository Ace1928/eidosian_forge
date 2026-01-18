from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
def _format_record(self, name, type, data, extra):
    if extra is None:
        extra = {}
    new_record = {}
    if type == RecordType.SRV:
        new_record = {'type': type, 'name': name, 'data': data, 'priority': 1, 'ttl': extra.get('ttl', 5), 'service': extra.get('service', ''), 'protocol': extra.get('protocol', ''), 'port': extra.get('port', ''), 'weight': extra.get('weight', '1')}
    else:
        new_record = {'type': type, 'name': name, 'data': data, 'ttl': extra.get('ttl', 5)}
    if type == RecordType.MX:
        new_record['priority'] = 1
    return new_record
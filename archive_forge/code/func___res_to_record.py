import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def __res_to_record(self, zone, record):
    if len(record['name']) == 0:
        name = None
    else:
        name = record['name']
    extra = {}
    extra['created'] = record['created']
    extra['modified'] = record['modified']
    extra['disabled'] = record['disabled']
    extra['ttl'] = record['ttl']
    extra['priority'] = record['prio']
    return Record(id=record['id'], name=name, type=record['type'], data=record['content'], zone=zone, driver=self.connection.driver, ttl=record['ttl'], extra=extra)
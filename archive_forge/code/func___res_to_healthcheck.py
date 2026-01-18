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
def __res_to_healthcheck(self, zone, healthcheck):
    return AuroraDNSHealthCheck(id=healthcheck['id'], type=healthcheck['type'], hostname=healthcheck['hostname'], ipaddress=healthcheck['ipaddress'], health=healthcheck['health'], threshold=healthcheck['threshold'], path=healthcheck['path'], interval=healthcheck['interval'], port=healthcheck['port'], enabled=healthcheck['enabled'], zone=zone, driver=self.connection.driver)
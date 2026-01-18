import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _iterate_records(params):
    url = '{}/zones/{}/dns_records'.format(API_BASE, zone.id)
    self.connection.set_context({'zone_id': zone.id})
    response = self.connection.request(url, params=params)
    items = response.object['result']
    records = [self._to_record(zone, item) for item in items]
    return (response, records)
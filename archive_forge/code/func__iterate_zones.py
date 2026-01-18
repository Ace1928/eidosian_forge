import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _iterate_zones(params):
    url = '{}/zones'.format(API_BASE)
    response = self.connection.request(url, params=params)
    items = response.object['result']
    zones = [self._to_zone(item) for item in items]
    return (response, zones)
import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _normalize_record_data_from_api(self, type, data):
    """
        Normalize record data for special records so it's consistent with
        the Libcloud API.
        """
    if not data:
        return data
    if type == RecordType.CAA:
        data = data.replace('\t', ' ')
    return data
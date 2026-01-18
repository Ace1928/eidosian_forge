import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class BaseDNSConnection:
    host = API_HOST
    secure = True
    responseCls = CloudFlareDNSResponse

    def encode_data(self, data):
        return json.dumps(data)
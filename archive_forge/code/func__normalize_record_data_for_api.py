import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _normalize_record_data_for_api(self, type, data):
    """
        Normalize record data for "special" records such as CAA and SSHFP
        so it can be used with the CloudFlare API.

        Keep ind mind that value for SSHFP record type onluy needs to be
        normalized for the create / update operations.

        On list operation (aka response), actual value is returned
        normally in the "content" attribute.
        """
    cf_data = {}
    if not data:
        return (data, cf_data)
    if type == RecordType.CAA:
        data = data.replace(' ', '\t')
    elif type == RecordType.SSHFP:
        _fp = data.split(' ')
        cf_data = {'algorithm': _fp[0], 'type': _fp[1], 'fingerprint': _fp[2]}
        data = None
    return (data, cf_data)
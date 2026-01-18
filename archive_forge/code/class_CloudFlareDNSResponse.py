import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class CloudFlareDNSResponse(JsonResponse):
    exceptions = {9103: (InvalidCredsError, []), 1001: (ZoneDoesNotExistError, ['zone_id']), 1061: (ZoneAlreadyExistsError, ['zone_id']), 1002: (RecordDoesNotExistError, ['record_id']), 81053: (RecordAlreadyExistsError, ['record_id']), 81057: (RecordAlreadyExistsError, []), 81058: (RecordAlreadyExistsError, ['record_id'])}

    def success(self):
        body = self.parse_body()
        is_success = body.get('success', False)
        return is_success

    def parse_error(self):
        body = self.parse_body()
        errors = body.get('errors', [])
        for error in errors:
            error_chain = error.get('error_chain', [])
            error_chain_errors = []
            for chain_error in error_chain:
                error_chain_errors.append('%s: %s' % (chain_error.get('code', 'unknown'), chain_error.get('message', '')))
            try:
                exception_class, context = self.exceptions[error['code']]
            except KeyError:
                exception_class, context = (LibcloudError, [])
            kwargs = {'value': '{}: {} (error chain: {})'.format(error['code'], error['message'], ', '.join(error_chain_errors)), 'driver': self.connection.driver}
            if error['code'] == 81057:
                kwargs['record_id'] = 'unknown'
            merge_valid_keys(kwargs, context, self.connection.context)
            raise exception_class(**kwargs)
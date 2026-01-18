import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def _validate_record(self, record_id, name, record_type, data, extra):
    if len(data) > 1024:
        raise RecordError('Record data must be <= 1024 characters', driver=self, record_id=record_id)
    if type == 'MX' or type == RecordType.MX:
        if extra is None or 'priority' not in extra:
            raise RecordError('MX record must have a priority', driver=self, record_id=record_id)
    if extra is not None and '_other_records' in extra:
        for other_value in extra.get('_other_records', []):
            if len(other_value['data']) > 1024:
                raise RecordError('Record data must be <= 1024 characters', driver=self, record_id=record_id)
            if type == 'MX' or type == RecordType.MX:
                if other_value['extra'] is None or 'priority' not in other_value['extra']:
                    raise RecordError('MX record must have a priority', driver=self, record_id=record_id)
    if extra is not None and 'ttl' in extra:
        if extra['ttl'] < TTL_MIN:
            raise RecordError('TTL must be at least 300 seconds', driver=self, record_id=record_id)
        if extra['ttl'] > TTL_MAX:
            raise RecordError('TTL must not exceed 30 days', driver=self, record_id=record_id)
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _record_to_json(record, zone_id):
    result = {'name': record.prefix or '@', 'value': record.target, 'type': record.type, 'zone_id': zone_id}
    if record.ttl is not None:
        result['ttl'] = record.ttl
    return result
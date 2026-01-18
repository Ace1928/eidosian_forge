from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _create_record_from_json(source, type=None, has_id=True):
    source = dict(source)
    result = DNSRecord()
    if has_id:
        result.id = source.pop('id')
    result.type = source.pop('type', type)
    result.ttl = source.pop('ttl', None)
    name = source.pop('name', None)
    if name == '@':
        name = None
    result.prefix = name
    result.target = source.pop('value')
    source.pop('zone_id', None)
    result.extra.update(source)
    return result
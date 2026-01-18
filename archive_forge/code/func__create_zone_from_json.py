from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible_collections.community.dns.plugins.module_utils.argspec import (
from ansible_collections.community.dns.plugins.module_utils.json_api_helper import (
from ansible_collections.community.dns.plugins.module_utils.provider import (
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _create_zone_from_json(source):
    zone = DNSZone(source['name'])
    zone.id = source['id']
    info = source.copy()
    info.pop('name')
    info.pop('id')
    if 'legacy_ns' in info:
        info['legacy_ns'] = sorted(info['legacy_ns'])
    zone.info = info
    return zone
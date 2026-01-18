from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def _get_zone_id(self, zone=None):
    if not zone:
        zone = self.zone
    zones = self.get_zones(zone)
    if len(zones) > 1:
        self.module.fail_json(msg='More than one zone matches {0}'.format(zone))
    if len(zones) < 1:
        self.module.fail_json(msg='No zone found with name {0}'.format(zone))
    return zones[0]['id']
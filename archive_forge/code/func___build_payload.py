from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __build_payload(self):
    payload = dict(data=self.module.params.get('data'), flags=self.module.params.get('flags'), name=self.module.params.get('name'), port=self.module.params.get('port'), priority=self.module.params.get('priority'), type=self.module.params.get('type'), tag=self.module.params.get('tag'), ttl=self.module.params.get('ttl'), weight=self.module.params.get('weight'))
    if payload['type'] != 'TXT' and payload['data']:
        payload['data'] = payload['data'].lower()
    if payload['data'] == self.domain:
        payload['data'] = '@'
    return payload
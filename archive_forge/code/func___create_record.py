from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __create_record(self):
    self.payload['data'] = self.__normalize_data()
    response = self.post('domains/%s/records' % self.domain, data=self.payload)
    status_code = response.status_code
    json = response.json
    if status_code == 201:
        changed = True
        return (changed, json['domain_record'])
    else:
        self.module.fail_json(msg='Error creating domain record [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
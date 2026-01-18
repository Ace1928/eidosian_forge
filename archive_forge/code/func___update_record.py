from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __update_record(self, record_id):
    self.payload['data'] = self.__normalize_data()
    record = self.__find_record_by_id(record_id)
    if record:
        response = self.put('domains/%(domain)s/records/%(record_id)s' % {'domain': self.domain, 'record_id': record_id}, data=self.payload)
        status_code = response.status_code
        json = response.json
        if status_code == 200:
            changed = True
            return (changed, json['domain_record'])
        else:
            self.module.fail_json(msg='Error updating domain record [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
    else:
        self.module.fail_json(msg='Error updating domain record. Record does not exist. [%s]' % record_id)
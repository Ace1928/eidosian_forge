from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __get_all_records(self):
    records = []
    page = 1
    while True:
        response = self.get('domains/%(domain)s/records?page=%(page)s' % {'domain': self.domain, 'page': page})
        status_code = response.status_code
        json = response.json
        if status_code != 200:
            self.module.fail_json(msg='Error getting domain records [%(status_code)s: %(json)s]' % {'status_code': status_code, 'json': json})
        for record in json['domain_records']:
            records.append(dict([(str(k), v) for k, v in record.items()]))
        if 'pages' in json['links'] and 'next' in json['links']['pages']:
            page += 1
        else:
            break
    return records
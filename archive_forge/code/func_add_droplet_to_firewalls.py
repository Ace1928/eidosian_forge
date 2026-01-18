from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def add_droplet_to_firewalls(self):
    changed = False
    rule = self.get_firewall_by_name()
    if rule is None:
        err = 'Failed to find firewalls: {0}'.format(self.module.params['firewall'])
        return err
    json_data = self.get_droplet()
    if json_data is not None:
        request_params = {}
        droplet = json_data.get('droplet', None)
        droplet_id = droplet.get('id', None)
        request_params['droplet_ids'] = [droplet_id]
        for firewall in rule:
            if droplet_id not in rule[firewall]['droplet_ids']:
                response = self.rest.post('firewalls/{0}/droplets'.format(rule[firewall]['id']), data=request_params)
                json_data = response.json
                status_code = response.status_code
                if status_code != 204:
                    err = 'Failed to add droplet {0} to firewall {1}'.format(droplet_id, rule[firewall]['id'])
                    return (err, changed)
                changed = True
    return (None, changed)
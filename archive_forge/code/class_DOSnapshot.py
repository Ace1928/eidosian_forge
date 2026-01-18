from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOSnapshot(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.wait = self.module.params.pop('wait', True)
        self.wait_timeout = self.module.params.pop('wait_timeout', 120)
        self.module.params.pop('oauth_token')
        self.snapshot_type = module.params['snapshot_type']
        self.snapshot_name = module.params['snapshot_name']
        self.snapshot_tags = module.params['snapshot_tags']
        self.snapshot_id = module.params['snapshot_id']
        self.volume_id = module.params['volume_id']

    def wait_finished(self):
        current_time = time.monotonic()
        end_time = current_time + self.wait_timeout
        while current_time < end_time:
            response = self.rest.get('actions/{0}'.format(str(self.action_id)))
            status = response.status_code
            if status != 200:
                self.module.fail_json(msg='Unable to find action {0}, please file a bug'.format(str(self.action_id)))
            json = response.json
            if json['action']['status'] == 'completed':
                return json
            time.sleep(10)
        self.module.fail_json(msg='Timed out waiting for snapshot, action {0}'.format(str(self.action_id)))

    def create(self):
        if self.module.check_mode:
            return self.module.exit_json(changed=True)
        if self.snapshot_type == 'droplet':
            droplet_id = self.module.params['droplet_id']
            data = {'type': 'snapshot'}
            if self.snapshot_name is not None:
                data['name'] = self.snapshot_name
            response = self.rest.post('droplets/{0}/actions'.format(str(droplet_id)), data=data)
            status = response.status_code
            json = response.json
            if status == 201:
                self.action_id = json['action']['id']
                if self.wait:
                    json = self.wait_finished()
                    self.module.exit_json(changed=True, msg='Created snapshot, action {0}'.format(self.action_id), data=json['action'])
                self.module.exit_json(changed=True, msg='Created snapshot, action {0}'.format(self.action_id), data=json['action'])
            else:
                self.module.fail_json(changed=False, msg='Failed to create snapshot: {0}'.format(json['message']))
        elif self.snapshot_type == 'volume':
            data = {'name': self.snapshot_name, 'tags': self.snapshot_tags}
            response = self.rest.post('volumes/{0}/snapshots'.format(str(self.volume_id)), data=data)
            status = response.status_code
            json = response.json
            if status == 201:
                self.module.exit_json(changed=True, msg='Created snapshot, snapshot {0}'.format(json['snapshot']['id']), data=json['snapshot'])
            else:
                self.module.fail_json(changed=False, msg='Failed to create snapshot: {0}'.format(json['message']))

    def delete(self):
        if self.module.check_mode:
            return self.module.exit_json(changed=True)
        response = self.rest.delete('snapshots/{0}'.format(str(self.snapshot_id)))
        status = response.status_code
        if status == 204:
            self.module.exit_json(changed=True, msg='Deleted snapshot {0}'.format(str(self.snapshot_id)))
        else:
            json = response.json
            self.module.fail_json(changed=False, msg='Failed to delete snapshot {0}: {1}'.format(self.snapshot_id, json['message']))
from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOBlockStorage(object):

    def __init__(self, module):
        self.module = module
        self.rest = DigitalOceanHelper(module)
        if self.module.params.get('project_name'):
            self.projects = DigitalOceanProjects(module, self.rest)

    def get_key_or_fail(self, k):
        v = self.module.params[k]
        if v is None:
            self.module.fail_json(msg='Unable to load %s' % k)
        return v

    def poll_action_for_complete_status(self, action_id):
        url = 'actions/{0}'.format(action_id)
        end_time = time.monotonic() + self.module.params['timeout']
        while time.monotonic() < end_time:
            time.sleep(10)
            response = self.rest.get(url)
            status = response.status_code
            json = response.json
            if status == 200:
                if json['action']['status'] == 'completed':
                    return True
                elif json['action']['status'] == 'errored':
                    raise DOBlockStorageException(json['message'])
        raise DOBlockStorageException('Unable to reach the DigitalOcean API at %s' % self.module.params.get('baseurl'))

    def get_block_storage_by_name(self, volume_name, region):
        url = 'volumes?name={0}&region={1}'.format(volume_name, region)
        resp = self.rest.get(url)
        if resp.status_code != 200:
            raise DOBlockStorageException(resp.json['message'])
        volumes = resp.json['volumes']
        if not volumes:
            return None
        return volumes[0]

    def get_attached_droplet_ID(self, volume_name, region):
        volume = self.get_block_storage_by_name(volume_name, region)
        if not volume or not volume['droplet_ids']:
            return None
        return volume['droplet_ids'][0]

    def attach_detach_block_storage(self, method, volume_name, region, droplet_id):
        data = {'type': method, 'volume_name': volume_name, 'region': region, 'droplet_id': droplet_id}
        response = self.rest.post('volumes/actions', data=data)
        status = response.status_code
        json = response.json
        if status == 202:
            return self.poll_action_for_complete_status(json['action']['id'])
        elif status == 200:
            return True
        elif status == 404 and method == 'detach':
            return False
        elif status == 422:
            return False
        else:
            raise DOBlockStorageException(json['message'])

    def resize_block_storage(self, volume_name, region, desired_size):
        if not desired_size:
            return False
        volume = self.get_block_storage_by_name(volume_name, region)
        if volume['size_gigabytes'] == desired_size:
            return False
        data = {'type': 'resize', 'size_gigabytes': desired_size}
        resp = self.rest.post('volumes/{0}/actions'.format(volume['id']), data=data)
        if resp.status_code == 202:
            return self.poll_action_for_complete_status(resp.json['action']['id'])
        else:
            raise DOBlockStorageException(resp.json['message'])

    def create_block_storage(self):
        volume_name = self.get_key_or_fail('volume_name')
        snapshot_id = self.module.params['snapshot_id']
        if snapshot_id:
            self.module.params['block_size'] = None
            self.module.params['region'] = None
            block_size = None
            region = None
        else:
            block_size = self.get_key_or_fail('block_size')
            region = self.get_key_or_fail('region')
        description = self.module.params['description']
        data = {'size_gigabytes': block_size, 'name': volume_name, 'description': description, 'region': region, 'snapshot_id': snapshot_id}
        response = self.rest.post('volumes', data=data)
        status = response.status_code
        json = response.json
        if status == 201:
            project_name = self.module.params.get('project_name')
            if project_name:
                urn = 'do:volume:{0}'.format(json['volume']['id'])
                assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
                self.module.exit_json(changed=True, id=json['volume']['id'], msg=error_message, assign_status=assign_status, resources=resources)
            else:
                self.module.exit_json(changed=True, id=json['volume']['id'])
        elif status == 409 and json['id'] == 'conflict':
            resized = self.resize_block_storage(volume_name, region, block_size)
            self.module.exit_json(changed=resized)
        else:
            raise DOBlockStorageException(json['message'])

    def delete_block_storage(self):
        volume_name = self.get_key_or_fail('volume_name')
        region = self.get_key_or_fail('region')
        url = 'volumes?name={0}&region={1}'.format(volume_name, region)
        attached_droplet_id = self.get_attached_droplet_ID(volume_name, region)
        if attached_droplet_id is not None:
            self.attach_detach_block_storage('detach', volume_name, region, attached_droplet_id)
        response = self.rest.delete(url)
        status = response.status_code
        json = response.json
        if status == 204:
            self.module.exit_json(changed=True)
        elif status == 404:
            self.module.exit_json(changed=False)
        else:
            raise DOBlockStorageException(json['message'])

    def attach_block_storage(self):
        volume_name = self.get_key_or_fail('volume_name')
        region = self.get_key_or_fail('region')
        droplet_id = self.get_key_or_fail('droplet_id')
        attached_droplet_id = self.get_attached_droplet_ID(volume_name, region)
        if attached_droplet_id is not None:
            if attached_droplet_id == droplet_id:
                self.module.exit_json(changed=False)
            else:
                self.attach_detach_block_storage('detach', volume_name, region, attached_droplet_id)
        changed_status = self.attach_detach_block_storage('attach', volume_name, region, droplet_id)
        self.module.exit_json(changed=changed_status)

    def detach_block_storage(self):
        volume_name = self.get_key_or_fail('volume_name')
        region = self.get_key_or_fail('region')
        droplet_id = self.get_key_or_fail('droplet_id')
        changed_status = self.attach_detach_block_storage('detach', volume_name, region, droplet_id)
        self.module.exit_json(changed=changed_status)
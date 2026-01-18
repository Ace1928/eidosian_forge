from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DODroplet(object):
    failure_message = {'empty_response': 'Empty response from the DigitalOcean API; please try again or open a bug if it never succeeds.', 'resizing_off': 'Droplet must be off prior to resizing: https://docs.digitalocean.com/reference/api/api-reference/#operation/post_droplet_action', 'unexpected': 'Unexpected error [{0}]; please file a bug: https://github.com/ansible-collections/community.digitalocean/issues', 'support_action': 'Error status on Droplet action [{0}], please try again or contact DigitalOcean support: https://docs.digitalocean.com/support/', 'failed_to': 'Failed to {0} {1} [HTTP {2}: {3}]'}

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.wait = self.module.params.pop('wait', True)
        self.wait_timeout = self.module.params.pop('wait_timeout', 120)
        self.unique_name = self.module.params.pop('unique_name', False)
        self.module.params.pop('oauth_token')
        self.id = None
        self.name = None
        self.size = None
        self.status = None
        if self.module.params.get('project_name'):
            self.projects = DigitalOceanProjects(module, self.rest)
        self.firewalls = self.get_firewalls()
        self.sleep_interval = self.module.params.pop('sleep_interval', 10)
        if self.wait:
            if self.sleep_interval > self.wait_timeout:
                self.module.fail_json(msg='Sleep interval {0} should be less than {1}'.format(self.sleep_interval, self.wait_timeout))
            if self.sleep_interval <= 0:
                self.module.fail_json(msg='Sleep interval {0} should be greater than zero'.format(self.sleep_interval))

    def get_firewalls(self):
        response = self.rest.get('firewalls')
        status_code = response.status_code
        json_data = response.json
        if status_code != 200:
            self.module.fail_json(msg='Failed to get firewalls', data=json_data)
        return self.rest.get_paginated_data(base_url='firewalls?', data_key_name='firewalls')

    def get_firewall_by_name(self):
        rule = {}
        item = 0
        for firewall in self.firewalls:
            for firewall_name in self.module.params['firewall']:
                if firewall_name in firewall['name']:
                    rule[item] = {}
                    rule[item].update(firewall)
                    item += 1
        if len(rule) > 0:
            return rule
        return None

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

    def remove_droplet_from_firewalls(self):
        changed = False
        json_data = self.get_droplet()
        if json_data is not None:
            request_params = {}
            droplet = json_data.get('droplet', None)
            droplet_id = droplet.get('id', None)
            request_params['droplet_ids'] = [droplet_id]
            for firewall in self.firewalls:
                if firewall['name'] not in self.module.params['firewall'] and droplet_id in firewall['droplet_ids']:
                    response = self.rest.delete('firewalls/{0}/droplets'.format(firewall['id']), data=request_params)
                    json_data = response.json
                    status_code = response.status_code
                    if status_code != 204:
                        err = 'Failed to remove droplet {0} from firewall {1}'.format(droplet_id, firewall['id'])
                        return (err, changed)
                    changed = True
        return (None, changed)

    def get_by_id(self, droplet_id):
        if not droplet_id:
            return None
        response = self.rest.get('droplets/{0}'.format(droplet_id))
        status_code = response.status_code
        json_data = response.json
        if json_data is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['empty_response'])
        else:
            if status_code == 200:
                droplet = json_data.get('droplet', None)
                if droplet is not None:
                    self.id = droplet.get('id', None)
                    self.name = droplet.get('name', None)
                    self.size = droplet.get('size_slug', None)
                    self.status = droplet.get('status', None)
                return json_data
            return None

    def get_by_name(self, droplet_name):
        if not droplet_name:
            return None
        page = 1
        while page is not None:
            response = self.rest.get('droplets?page={0}'.format(page))
            json_data = response.json
            status_code = response.status_code
            if json_data is None:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['empty_response'])
            elif status_code == 200:
                droplets = json_data.get('droplets', [])
                for droplet in droplets:
                    if droplet.get('name', None) == droplet_name:
                        self.id = droplet.get('id', None)
                        self.name = droplet.get('name', None)
                        self.size = droplet.get('size_slug', None)
                        self.status = droplet.get('status', None)
                        return {'droplet': droplet}
                if 'links' in json_data and 'pages' in json_data['links'] and ('next' in json_data['links']['pages']):
                    page += 1
                else:
                    page = None
        return None

    def get_addresses(self, data):
        """Expose IP addresses as their own property allowing users extend to additional tasks"""
        _data = data
        for k, v in data.items():
            setattr(self, k, v)
        networks = _data['droplet']['networks']
        for network in networks.get('v4', []):
            if network['type'] == 'public':
                _data['ip_address'] = network['ip_address']
            else:
                _data['private_ipv4_address'] = network['ip_address']
        for network in networks.get('v6', []):
            if network['type'] == 'public':
                _data['ipv6_address'] = network['ip_address']
            else:
                _data['private_ipv6_address'] = network['ip_address']
        return _data

    def get_droplet(self):
        json_data = self.get_by_id(self.module.params['id'])
        if not json_data and self.unique_name:
            json_data = self.get_by_name(self.module.params['name'])
        return json_data

    def resize_droplet(self, state, droplet_id):
        if self.status != 'off':
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['resizing_off'])
        self.wait_action(droplet_id, {'type': 'resize', 'disk': self.module.params['resize_disk'], 'size': self.module.params['size']})
        if state == 'active':
            self.ensure_power_on(droplet_id)
        json_data = self.get_droplet()
        droplet = json_data.get('droplet', None)
        if droplet is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet'))
        self.module.exit_json(changed=True, msg='Resized Droplet {0} ({1}) from {2} to {3}'.format(self.name, self.id, self.size, self.module.params['size']), data={'droplet': droplet})

    def wait_status(self, droplet_id, desired_statuses):
        end_time = time.monotonic() + self.wait_timeout
        while time.monotonic() < end_time:
            response = self.rest.get('droplets/{0}'.format(droplet_id))
            json_data = response.json
            status_code = response.status_code
            message = json_data.get('message', 'no error message')
            droplet = json_data.get('droplet', None)
            droplet_status = droplet.get('status', None) if droplet else None
            if droplet is None or droplet_status is None:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet or status'))
            if status_code >= 400:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('get', 'Droplet', status_code, message))
            if droplet_status in desired_statuses:
                return
            time.sleep(self.sleep_interval)
        self.module.fail_json(msg='Wait for Droplet [{0}] status timeout'.format(','.join(desired_statuses)))

    def wait_check_action(self, droplet_id, action_id):
        end_time = time.monotonic() + self.wait_timeout
        while time.monotonic() < end_time:
            response = self.rest.get('droplets/{0}/actions/{1}'.format(droplet_id, action_id))
            json_data = response.json
            status_code = response.status_code
            message = json_data.get('message', 'no error message')
            action = json_data.get('action', None)
            action_id = action.get('id', None)
            action_status = action.get('status', None)
            if action is None or action_id is None or action_status is None:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no action, ID, or status'))
            if status_code >= 400:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('get', 'action', status_code, message))
            if action_status == 'errored':
                self.module.fail_json(changed=True, msg=DODroplet.failure_message['support_action'].format(action_id))
            if action_status == 'completed':
                return
            time.sleep(self.sleep_interval)
        self.module.fail_json(msg='Wait for Droplet action timeout')

    def wait_action(self, droplet_id, desired_action_data):
        action_type = desired_action_data.get('type', 'undefined')
        response = self.rest.post('droplets/{0}/actions'.format(droplet_id), data=desired_action_data)
        json_data = response.json
        status_code = response.status_code
        message = json_data.get('message', 'no error message')
        action = json_data.get('action', None)
        action_id = action.get('id', None)
        action_status = action.get('status', None)
        if action is None or action_id is None or action_status is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no action, ID, or status'))
        if status_code >= 400:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('post', 'action', status_code, message))
        self.wait_check_action(droplet_id, action_id)

    def ensure_power_on(self, droplet_id):
        self.wait_status(droplet_id, ['active', 'off'])
        self.wait_action(droplet_id, {'type': 'power_on'})

    def ensure_power_off(self, droplet_id):
        self.wait_status(droplet_id, ['active'])
        self.wait_action(droplet_id, {'type': 'power_off'})

    def create(self, state):
        json_data = self.get_droplet()
        if json_data is not None:
            droplet = json_data.get('droplet', None)
            droplet_id = droplet.get('id', None)
            droplet_size = droplet.get('size_slug', None)
            if droplet_id is None or droplet_size is None:
                self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet ID or size'))
            if self.module.params['firewall'] is not None:
                firewall_changed = False
                if len(self.module.params['firewall']) > 0:
                    firewall_add, add_changed = self.add_droplet_to_firewalls()
                    if firewall_add is not None:
                        self.module.fail_json(changed=False, msg=firewall_add, data={'droplet': droplet, 'firewall': firewall_add})
                    firewall_changed = firewall_changed or add_changed
                firewall_remove, remove_changed = self.remove_droplet_from_firewalls()
                if firewall_remove is not None:
                    self.module.fail_json(changed=False, msg=firewall_remove, data={'droplet': droplet, 'firewall': firewall_remove})
                firewall_changed = firewall_changed or remove_changed
                self.module.exit_json(changed=firewall_changed, data={'droplet': droplet})
            if self.module.check_mode:
                self.module.exit_json(changed=False)
            if droplet_size != self.module.params.get('size', None):
                self.resize_droplet(state, droplet_id)
            droplet_data = self.get_addresses(json_data)
            droplet_id = droplet.get('id', None)
            droplet_status = droplet.get('status', None)
            if droplet_id is not None and droplet_status is not None:
                if state == 'active' and droplet_status != 'active':
                    self.ensure_power_on(droplet_id)
                    json_data = self.get_droplet()
                    droplet = json_data.get('droplet', droplet)
                    self.module.exit_json(changed=True, data={'droplet': droplet})
                elif state == 'inactive' and droplet_status != 'off':
                    self.ensure_power_off(droplet_id)
                    json_data = self.get_droplet()
                    droplet = json_data.get('droplet', droplet)
                    self.module.exit_json(changed=True, data={'droplet': droplet})
                else:
                    self.module.exit_json(changed=False, data={'droplet': droplet})
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        request_params = dict(self.module.params)
        del request_params['id']
        response = self.rest.post('droplets', data=request_params)
        json_data = response.json
        status_code = response.status_code
        message = json_data.get('message', 'no error message')
        droplet = json_data.get('droplet', None)
        if status_code != 202:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('create', 'Droplet', status_code, message))
        droplet_id = droplet.get('id', None)
        if droplet is None or droplet_id is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet or ID'))
        if status_code >= 400:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['failed_to'].format('create', 'Droplet', status_code, message))
        if self.wait:
            if state == 'present' or state == 'active':
                self.ensure_power_on(droplet_id)
            if state == 'inactive':
                self.ensure_power_off(droplet_id)
        elif state == 'inactive':
            self.ensure_power_off(droplet_id)
        if self.wait:
            json_data = self.get_by_id(droplet_id)
            if json_data:
                droplet = json_data.get('droplet', droplet)
        project_name = self.module.params.get('project_name')
        if project_name:
            urn = 'do:droplet:{0}'.format(droplet_id)
            assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
            self.module.exit_json(changed=True, data={'droplet': droplet}, msg=error_message, assign_status=assign_status, resources=resources)
        if self.module.params['firewall'] is not None:
            firewall_add = self.add_droplet_to_firewalls()
            if firewall_add is not None:
                self.module.fail_json(changed=False, msg=firewall_add, data={'droplet': droplet, 'firewall': firewall_add})
            firewall_remove = self.remove_droplet_from_firewalls()
            if firewall_remove is not None:
                self.module.fail_json(changed=False, msg=firewall_remove, data={'droplet': droplet, 'firewall': firewall_remove})
            self.module.exit_json(changed=True, data={'droplet': droplet})
        self.module.exit_json(changed=True, data={'droplet': droplet})

    def delete(self):
        if not self.module.params['id'] and (not self.unique_name):
            self.module.fail_json(changed=False, msg='id must be set or unique_name must be true for deletes')
        json_data = self.get_droplet()
        if json_data is None:
            self.module.exit_json(changed=False, msg='Droplet not found')
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        droplet = json_data.get('droplet', None)
        droplet_id = droplet.get('id', None)
        droplet_name = droplet.get('name', None)
        if droplet is None or droplet_id is None:
            self.module.fail_json(changed=False, msg=DODroplet.failure_message['unexpected'].format('no Droplet, name, or ID'))
        response = self.rest.delete('droplets/{0}'.format(droplet_id))
        json_data = response.json
        status_code = response.status_code
        if status_code == 204:
            self.module.exit_json(changed=True, msg='Droplet {0} ({1}) deleted'.format(droplet_name, droplet_id))
        else:
            self.module.fail_json(changed=False, msg='Failed to delete Droplet {0} ({1})'.format(droplet_name, droplet_id))
from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
class AnsibleCloudscaleBase(AnsibleCloudscaleApi):

    def __init__(self, module, resource_name='', resource_key_uuid='uuid', resource_key_name='name', resource_create_param_keys=None, resource_update_param_keys=None):
        super(AnsibleCloudscaleBase, self).__init__(module)
        self._result = {'changed': False, 'diff': dict(before=dict(), after=dict())}
        self._resource_data = dict()
        self.resource_key_uuid = resource_key_uuid
        self.resource_key_name = resource_key_name
        self.resource_name = resource_name
        self.resource_create_param_keys = resource_create_param_keys or ['name']
        self.resource_update_param_keys = resource_update_param_keys or ['name']
        self.use_tag_for_name = False
        self.resource_name_tag = 'ansible_name'
        self.query_constraint_keys = []

    def pre_transform(self, resource):
        return resource

    def init_resource(self):
        return {'state': 'absent', self.resource_key_uuid: self._module.params.get(self.resource_key_uuid) or self._resource_data.get(self.resource_key_uuid), self.resource_key_name: self._module.params.get(self.resource_key_name) or self._resource_data.get(self.resource_key_name)}

    def query(self):
        self._resource_data = self.init_resource()
        uuid = self._module.params[self.resource_key_uuid]
        if uuid is not None:
            if '/' in uuid:
                uuid = uuid.split('/')[0]
            resource = self._get('%s/%s' % (self.resource_name, uuid))
            if resource:
                self._resource_data = resource
                self._resource_data['state'] = 'present'
        else:
            name = self._module.params[self.resource_key_name]
            if self.use_tag_for_name:
                resources = self._get('%s?tag:%s=%s' % (self.resource_name, self.resource_name_tag, name))
            else:
                resources = self._get('%s' % self.resource_name)
            matching = []
            for resource in resources:
                if self.use_tag_for_name:
                    resource[self.resource_key_name] = resource['tags'].get(self.resource_name_tag)
                for constraint_key in self.query_constraint_keys:
                    if self._module.params[constraint_key] is not None:
                        if constraint_key == 'zone':
                            resource_value = resource['zone']['slug']
                        else:
                            resource_value = resource[constraint_key]
                        if resource_value != self._module.params[constraint_key]:
                            break
                else:
                    if resource[self.resource_key_name] == name:
                        matching.append(resource)
            if len(matching) > 1:
                self._module.fail_json(msg="More than one %s resource with '%s' exists: %s. Use the '%s' parameter to identify the resource." % (self.resource_name, self.resource_key_name, name, self.resource_key_uuid))
            elif len(matching) == 1:
                self._resource_data = matching[0]
                self._resource_data['state'] = 'present'
        return self.pre_transform(self._resource_data)

    def create(self, resource, data=None):
        uuid = self._module.params.get(self.resource_key_uuid)
        if uuid is not None:
            self._module.fail_json(msg="The resource with UUID '%s' was not found and we would create a new one with different UUID, this is probably not want you have asked for." % uuid)
        self._result['changed'] = True
        if not data:
            data = dict()
        for param in self.resource_create_param_keys:
            data[param] = self._module.params.get(param)
        self._result['diff']['before'] = deepcopy(resource)
        self._result['diff']['after'] = deepcopy(resource)
        self._result['diff']['after'].update(deepcopy(data))
        self._result['diff']['after'].update({'state': 'present'})
        if not self._module.check_mode:
            resource = self._post(self.resource_name, data)
            resource = self.pre_transform(resource)
            resource['state'] = 'present'
        return resource

    def wait_for_state(self, check_parameter, allowed_states):
        start = datetime.now()
        timeout = self._module.params['api_timeout'] * 2
        while datetime.now() - start < timedelta(seconds=timeout):
            info = self.query()
            if info.get(check_parameter) in allowed_states:
                return info
            sleep(1)
        name_uuid = info.get('name') or self._module.params.get('name') or self._module.params.get('uuid')
        msg = 'Timeout while waiting for a state change for resource %s to states %s' % (name_uuid, allowed_states)
        self._module.fail_json(msg=msg)

    def update(self, resource):
        updated = False
        for param in self.resource_update_param_keys:
            updated = self._param_updated(param, resource) or updated
        if updated and (not self._module.check_mode):
            resource = self.query()
        return resource

    def present(self):
        resource = self.query()
        if self.use_tag_for_name:
            name_tag_value = self._module.params[self.resource_key_name] or resource.get('tags', dict()).get(self.resource_name_tag)
            if name_tag_value:
                self._module.params['tags'] = self._module.params['tags'] or dict()
                self._module.params['tags'].update({self.resource_name_tag: name_tag_value})
        if resource['state'] == 'absent':
            resource = self.create(resource)
        else:
            resource = self.update(resource)
        return self.get_result(resource)

    def absent(self):
        resource = self.query()
        if resource['state'] != 'absent':
            self._result['changed'] = True
            self._result['diff']['before'] = deepcopy(resource)
            self._result['diff']['after'] = self.init_resource()
            if not self._module.check_mode:
                href = resource.get('href')
                if not href:
                    self._module.fail_json(msg='Unable to delete %s, no href found.')
                self._delete(href)
                resource['state'] = 'absent'
        return self.get_result(resource)

    def find_difference(self, key, resource, param):
        is_different = False
        if isinstance(resource[key], dict) and 'href' in resource[key]:
            uuid = resource[key].get('href', '').split('/')[-1]
            if param != uuid:
                is_different = True
        elif param != resource[key]:
            is_different = True
        return is_different

    def _param_updated(self, key, resource):
        param = self._module.params.get(key)
        if param is None:
            return False
        if not resource or key not in resource:
            return False
        is_different = self.find_difference(key, resource, param)
        if is_different:
            self._result['changed'] = True
            patch_data = {key: param}
            self._result['diff']['before'].update({key: resource[key]})
            self._result['diff']['after'].update(patch_data)
            if not self._module.check_mode:
                href = resource.get('href')
                if not href:
                    self._module.fail_json(msg='Unable to update %s, no href found.' % key)
                self._patch(href, patch_data)
                return True
        return False

    def get_result(self, resource):
        if resource:
            for k, v in resource.items():
                self._result[k] = v
            if self.use_tag_for_name:
                self._result['name'] = self._result.get('tags', dict()).pop(self.resource_name_tag, None)
        return self._result
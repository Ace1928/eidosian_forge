from __future__ import absolute_import, division, print_function
import urllib
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrReservedIp(AnsibleVultr):

    def configure(self):
        self.instance_id = self.get_instance_id()

    def get_instance_id(self):
        instance_id = self.module.params['instance_id']
        if instance_id is not None:
            return instance_id
        instance_name = self.module.params['instance_name']
        if instance_name is not None:
            if len(instance_name) == 0:
                return ''
            try:
                label = urllib.quote(instance_name)
            except AttributeError:
                label = urllib.parse.quote(instance_name)
            resources = self.api_query(path='/instances?label=%s' % label) or dict()
            if not resources or not resources['instances']:
                self.module.fail_json(msg='No instance with name found: %s' % instance_name)
            if len(resources['instances']) > 1:
                self.module.fail_json(msg='More then one instance with name found: %s' % instance_name)
            return resources['instances'][0]['id']

    def query_list(self, path=None, result_key=None, query_params=None):
        resources = self.api_query(path=self.resource_path) or dict()
        resources_filtered = list()
        for resource in resources[self.ressource_result_key_plural]:
            if resource['ip_type'] != self.module.params['ip_type']:
                continue
            if resource['region'] != self.module.params['region']:
                continue
            resources_filtered.append(resource)
        return resources_filtered

    def create(self):
        resource = super().create() or dict()
        if resource and self.instance_id:
            if not self.module.check_mode:
                self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], 'attach'), method='POST', data=dict(instance_id=self.instance_id))
                resource = self.query_by_id(resource_id=resource[self.resource_key_id])
        return resource

    def update(self, resource):
        if self.instance_id is None:
            return resource
        elif resource['instance_id'] and (not self.instance_id):
            self.result['changed'] = True
            if not self.module.check_mode:
                self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], 'detach'), method='POST', data=dict(instance_id=self.instance_id))
                resource = self.query_by_id(resource_id=resource[self.resource_key_id])
        elif self.instance_id and resource['instance_id'] != self.instance_id:
            self.result['changed'] = True
            if not self.module.check_mode:
                self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], 'attach'), method='POST', data=dict(instance_id=self.instance_id))
                resource = self.query_by_id(resource_id=resource[self.resource_key_id])
        return resource
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleLoadBalancerPoolMember(AnsibleCloudscaleBase):

    def __init__(self, module):
        super(AnsibleCloudscaleLoadBalancerPoolMember, self).__init__(module, resource_name='load-balancers/pools/%s/members' % module.params['load_balancer_pool'], resource_create_param_keys=['name', 'enabled', 'protocol_port', 'monitor_port', 'address', 'subnet', 'tags'], resource_update_param_keys=['name', 'enabled', 'tags'])

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
            if resources is None:
                self._module.fail_json(msg='The load balancer pool %s does not exist.' % (self.resource_name,))
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
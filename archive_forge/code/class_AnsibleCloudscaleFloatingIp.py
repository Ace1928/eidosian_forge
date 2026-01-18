from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleFloatingIp(AnsibleCloudscaleBase):

    def __init__(self, module):
        super(AnsibleCloudscaleFloatingIp, self).__init__(module=module, resource_key_uuid='network', resource_name='floating-ips', resource_create_param_keys=['ip_version', 'server', 'prefix_length', 'reverse_ptr', 'type', 'region', 'tags'], resource_update_param_keys=['server', 'reverse_ptr', 'tags'])
        self.use_tag_for_name = True
        self.query_constraint_keys = ['ip_version']

    def pre_transform(self, resource):
        if 'server' in resource and isinstance(resource['server'], dict):
            resource['server'] = resource['server']['uuid']
        return resource

    def create(self, resource):
        self._module.fail_on_missing_params(['ip_version', 'name'])
        return super(AnsibleCloudscaleFloatingIp, self).create(resource)

    def get_result(self, resource):
        network = resource.get('network')
        if network:
            self._result['ip'] = network.split('/')[0]
        return super(AnsibleCloudscaleFloatingIp, self).get_result(resource)
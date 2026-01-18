from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
class AnsibleCloudscaleLoadBalancerHealthMonitor(AnsibleCloudscaleBase):

    def __init__(self, module):
        super(AnsibleCloudscaleLoadBalancerHealthMonitor, self).__init__(module, resource_name='load-balancers/health-monitors', resource_key_name='pool', resource_create_param_keys=['pool', 'timeout_s', 'up_threshold', 'down_threshold', 'type', 'http', 'tags'], resource_update_param_keys=['delay_s', 'timeout_s', 'up_threshold', 'down_threshold', 'expected_codes', 'http', 'tags'])

    def query(self):
        self._resource_data = self.init_resource()
        resource_key_pool = 'pool'
        uuid = self._module.params[self.resource_key_uuid]
        pool = self._module.params[resource_key_pool]
        matching = []
        if uuid is not None:
            super().query()
        else:
            pool = self._module.params[resource_key_pool]
            if pool is not None:
                resources = self._get('%s' % self.resource_name)
                if resources:
                    for health_monitor in resources:
                        if health_monitor[resource_key_pool]['uuid'] == pool:
                            matching.append(health_monitor)
            if len(matching) > 1:
                self._module.fail_json(msg="More than one %s resource for pool '%s' exists." % (self.resource_name, resource_key_pool))
            elif len(matching) == 1:
                self._resource_data = matching[0]
                self._resource_data['state'] = 'present'
        return self.pre_transform(self._resource_data)

    def update(self, resource):
        updated = False
        for param in self.resource_update_param_keys:
            if param == 'http' and self._module.params.get('http') is not None:
                for subparam in ALLOWED_HTTP_POST_PARAMS:
                    updated = self._http_param_updated(subparam, resource) or updated
            else:
                updated = self._param_updated(param, resource) or updated
        if updated and (not self._module.check_mode):
            resource = self.query()
        return resource

    def _http_param_updated(self, key, resource):
        param_http = self._module.params.get('http')
        param = param_http[key]
        if param is None:
            return False
        if not resource or key not in resource['http']:
            return False
        is_different = self.find_http_difference(key, resource, param)
        if is_different:
            self._result['changed'] = True
            patch_data = {'http': {key: param}}
            before_data = {'http': {key: resource['http'][key]}}
            self._result['diff']['before'].update(before_data)
            self._result['diff']['after'].update(patch_data)
            if not self._module.check_mode:
                href = resource.get('href')
                if not href:
                    self._module.fail_json(msg='Unable to update %s, no href found.' % key)
                self._patch(href, patch_data)
                return True
        return False

    def find_http_difference(self, key, resource, param):
        is_different = False
        if param != resource['http'][key]:
            is_different = True
        return is_different
from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def _create_load_balancer_service(self):
    self.module.fail_on_missing_params(required_params=['protocol'])
    if self.module.params.get('protocol') == 'tcp':
        self.module.fail_on_missing_params(required_params=['destination_port'])
    params = {'protocol': self.module.params.get('protocol'), 'listen_port': self.module.params.get('listen_port'), 'proxyprotocol': self.module.params.get('proxyprotocol')}
    if self.module.params.get('destination_port'):
        params['destination_port'] = self.module.params.get('destination_port')
    if self.module.params.get('http'):
        params['http'] = self.__get_service_http(http_arg=self.module.params.get('http'))
    if self.module.params.get('health_check'):
        params['health_check'] = self.__get_service_health_checks(health_check=self.module.params.get('health_check'))
    if not self.module.check_mode:
        try:
            self.hcloud_load_balancer.add_service(LoadBalancerService(**params)).wait_until_finished(max_retries=1000)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_load_balancer()
    self._get_load_balancer_service()
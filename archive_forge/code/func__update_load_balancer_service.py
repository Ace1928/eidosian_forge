from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def _update_load_balancer_service(self):
    changed = False
    try:
        params = {'listen_port': self.module.params.get('listen_port')}
        if self.module.params.get('destination_port') is not None:
            if self.hcloud_load_balancer_service.destination_port != self.module.params.get('destination_port'):
                params['destination_port'] = self.module.params.get('destination_port')
                changed = True
        if self.module.params.get('protocol') is not None:
            if self.hcloud_load_balancer_service.protocol != self.module.params.get('protocol'):
                params['protocol'] = self.module.params.get('protocol')
                changed = True
        if self.module.params.get('proxyprotocol') is not None:
            if self.hcloud_load_balancer_service.proxyprotocol != self.module.params.get('proxyprotocol'):
                params['proxyprotocol'] = self.module.params.get('proxyprotocol')
                changed = True
        if self.module.params.get('http') is not None:
            params['http'] = self.__get_service_http(http_arg=self.module.params.get('http'))
            changed = True
        if self.module.params.get('health_check') is not None:
            params['health_check'] = self.__get_service_health_checks(health_check=self.module.params.get('health_check'))
            changed = True
        if not self.module.check_mode:
            self.hcloud_load_balancer.update_service(LoadBalancerService(**params)).wait_until_finished(max_retries=1000)
    except HCloudException as exception:
        self.fail_json_hcloud(exception)
    self._get_load_balancer()
    if changed:
        self._mark_as_changed()
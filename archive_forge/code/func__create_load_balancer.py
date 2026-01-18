from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def _create_load_balancer(self):
    self.module.fail_on_missing_params(required_params=['name', 'load_balancer_type'])
    try:
        params = {'name': self.module.params.get('name'), 'algorithm': LoadBalancerAlgorithm(type=self.module.params.get('algorithm', 'round_robin')), 'load_balancer_type': self.client.load_balancer_types.get_by_name(self.module.params.get('load_balancer_type')), 'labels': self.module.params.get('labels')}
        if self.module.params.get('location') is None and self.module.params.get('network_zone') is None:
            self.module.fail_json(msg='one of the following is required: location, network_zone')
        elif self.module.params.get('location') is not None and self.module.params.get('network_zone') is None:
            params['location'] = self.client.locations.get_by_name(self.module.params.get('location'))
        elif self.module.params.get('location') is None and self.module.params.get('network_zone') is not None:
            params['network_zone'] = self.module.params.get('network_zone')
        if not self.module.check_mode:
            resp = self.client.load_balancers.create(**params)
            resp.action.wait_until_finished(max_retries=1000)
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None:
                self._get_load_balancer()
                self.hcloud_load_balancer.change_protection(delete=delete_protection).wait_until_finished()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_load_balancer()
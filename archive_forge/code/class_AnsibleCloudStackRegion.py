from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackRegion(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackRegion, self).__init__(module)
        self.returns = {'endpoint': 'endpoint', 'gslbserviceenabled': 'gslb_service_enabled', 'portableipserviceenabled': 'portable_ip_service_enabled'}

    def get_region(self):
        id = self.module.params.get('id')
        regions = self.query_api('listRegions', id=id)
        if regions:
            return regions['region'][0]
        return None

    def present_region(self):
        region = self.get_region()
        if not region:
            region = self._create_region(region=region)
        else:
            region = self._update_region(region=region)
        return region

    def _create_region(self, region):
        self.result['changed'] = True
        args = {'id': self.module.params.get('id'), 'name': self.module.params.get('name'), 'endpoint': self.module.params.get('endpoint')}
        if not self.module.check_mode:
            res = self.query_api('addRegion', **args)
            region = res['region']
        return region

    def _update_region(self, region):
        args = {'id': self.module.params.get('id'), 'name': self.module.params.get('name'), 'endpoint': self.module.params.get('endpoint')}
        if self.has_changed(args, region):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateRegion', **args)
                region = res['region']
        return region

    def absent_region(self):
        region = self.get_region()
        if region:
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('removeRegion', id=region['id'])
        return region
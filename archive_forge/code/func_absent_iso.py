from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_iso(self):
    iso = self.get_iso()
    if iso:
        self.result['changed'] = True
        args = {'id': iso['id'], 'projectid': self.get_project('id')}
        if not self.module.params.get('cross_zones'):
            args['zoneid'] = self.get_zone(key='id')
        if not self.module.check_mode:
            res = self.query_api('deleteIso', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'iso')
    return iso
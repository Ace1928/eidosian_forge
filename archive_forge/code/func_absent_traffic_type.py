from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def absent_traffic_type(self):
    traffic_type = self.get_traffic_type()
    if traffic_type:
        args = {'id': traffic_type['id']}
        self.result['changed'] = True
        if not self.module.check_mode:
            resource = self.query_api('deleteTrafficType', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(resource, 'traffictype')
    return traffic_type
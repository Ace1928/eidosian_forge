from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackSnapshotPolicy(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackSnapshotPolicy, self).__init__(module)
        self.returns = {'schedule': 'schedule', 'timezone': 'time_zone', 'maxsnaps': 'max_snaps'}
        self.interval_types = {'hourly': 0, 'daily': 1, 'weekly': 2, 'monthly': 3}
        self.volume = None

    def get_interval_type(self):
        interval_type = self.module.params.get('interval_type')
        return self.interval_types[interval_type]

    def get_volume(self, key=None):
        if self.volume:
            return self._get_by_key(key, self.volume)
        args = {'name': self.module.params.get('volume'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'virtualmachineid': self.get_vm(key='id', filter_zone=False), 'type': self.module.params.get('volume_type')}
        volumes = self.query_api('listVolumes', **args)
        if volumes:
            if volumes['count'] > 1:
                device_id = self.module.params.get('device_id')
                if not device_id:
                    self.module.fail_json(msg="Found more then 1 volume: combine params 'vm', 'volume_type', 'device_id' and/or 'volume' to select the volume")
                else:
                    for v in volumes['volume']:
                        if v.get('deviceid') == device_id:
                            self.volume = v
                            return self._get_by_key(key, self.volume)
                    self.module.fail_json(msg='No volume found with device id %s' % device_id)
            self.volume = volumes['volume'][0]
            return self._get_by_key(key, self.volume)
        return None

    def get_snapshot_policy(self):
        args = {'volumeid': self.get_volume(key='id')}
        policies = self.query_api('listSnapshotPolicies', **args)
        if policies:
            for policy in policies['snapshotpolicy']:
                if policy['intervaltype'] == self.get_interval_type():
                    return policy
            return None

    def present_snapshot_policy(self):
        required_params = ['schedule']
        self.module.fail_on_missing_params(required_params=required_params)
        policy = self.get_snapshot_policy()
        args = {'id': policy.get('id') if policy else None, 'intervaltype': self.module.params.get('interval_type'), 'schedule': self.module.params.get('schedule'), 'maxsnaps': self.module.params.get('max_snaps'), 'timezone': self.module.params.get('time_zone'), 'volumeid': self.get_volume(key='id')}
        if not policy or (policy and self.has_changed(policy, args, only_keys=['schedule', 'maxsnaps', 'timezone'])):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('createSnapshotPolicy', **args)
                policy = res['snapshotpolicy']
        return policy

    def absent_snapshot_policy(self):
        policy = self.get_snapshot_policy()
        if policy:
            self.result['changed'] = True
            args = {'id': policy['id']}
            if not self.module.check_mode:
                self.query_api('deleteSnapshotPolicies', **args)
        return policy

    def get_result(self, resource):
        super(AnsibleCloudStackSnapshotPolicy, self).get_result(resource)
        if resource and 'intervaltype' in resource:
            for key, value in self.interval_types.items():
                if value == resource['intervaltype']:
                    self.result['interval_type'] = key
                    break
        volume = self.get_volume()
        if volume:
            volume_results = {'volume': volume.get('name'), 'zone': volume.get('zonename'), 'project': volume.get('project'), 'account': volume.get('account'), 'domain': volume.get('domain')}
            self.result.update(volume_results)
        return self.result
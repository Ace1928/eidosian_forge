from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackVmSnapshot(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackVmSnapshot, self).__init__(module)
        self.returns = {'type': 'type', 'current': 'current'}

    def get_snapshot(self):
        args = {'virtualmachineid': self.get_vm('id'), 'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id'), 'name': self.module.params.get('name')}
        snapshots = self.query_api('listVMSnapshot', **args)
        if snapshots:
            return snapshots['vmSnapshot'][0]
        return None

    def create_snapshot(self):
        snapshot = self.get_snapshot()
        if not snapshot:
            self.result['changed'] = True
            args = {'virtualmachineid': self.get_vm('id'), 'name': self.module.params.get('name'), 'description': self.module.params.get('description'), 'snapshotmemory': self.module.params.get('snapshot_memory')}
            if not self.module.check_mode:
                res = self.query_api('createVMSnapshot', **args)
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    snapshot = self.poll_job(res, 'vmsnapshot')
        if snapshot:
            snapshot = self.ensure_tags(resource=snapshot, resource_type='Snapshot')
        return snapshot

    def remove_snapshot(self):
        snapshot = self.get_snapshot()
        if snapshot:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('deleteVMSnapshot', vmsnapshotid=snapshot['id'])
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    res = self.poll_job(res, 'vmsnapshot')
        return snapshot

    def revert_vm_to_snapshot(self):
        snapshot = self.get_snapshot()
        if snapshot:
            self.result['changed'] = True
            if snapshot['state'] != 'Ready':
                self.module.fail_json(msg="snapshot state is '%s', not ready, could not revert VM" % snapshot['state'])
            if not self.module.check_mode:
                res = self.query_api('revertToVMSnapshot', vmsnapshotid=snapshot['id'])
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    res = self.poll_job(res, 'vmsnapshot')
            return snapshot
        self.module.fail_json(msg='snapshot not found, could not revert VM')
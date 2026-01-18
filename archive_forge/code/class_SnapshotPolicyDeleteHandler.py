from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyDeleteHandler:

    def handle(self, con_object, con_params, snapshot_policy_details):
        if con_params['state'] == 'absent' and snapshot_policy_details:
            snapshot_policy_details = con_object.delete_snapshot_policy(snap_pol_id=snapshot_policy_details.get('id'))
            con_object.result['changed'] = True
        SnapshotPolicyExitHandler().handle(con_object, snapshot_policy_details)
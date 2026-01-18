from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyModifyHandler:

    def handle(self, con_object, con_params, snapshot_policy_details, auto_snapshot_creation_cadence_in_min):
        modify_dict = {}
        if con_params['state'] == 'present' and snapshot_policy_details:
            modify_dict = con_object.to_modify(snap_pol_details=snapshot_policy_details, new_name=con_params['new_name'], auto_snapshot_creation_cadence_in_min=auto_snapshot_creation_cadence_in_min, num_of_retained_snapshots_per_level=con_params['num_of_retained_snapshots_per_level'])
            msg = f'Parameters to be modified are as follows: {str(modify_dict)}'
            LOG.info(msg)
        if modify_dict and con_params['state'] == 'present':
            con_object.result['changed'] = con_object.modify_snapshot_policy(snap_pol_details=snapshot_policy_details, modify_dict=modify_dict)
            snapshot_policy_details = con_object.get_snapshot_policy(snap_pol_id=snapshot_policy_details.get('id'))
        SnapshotPolicySourceVolumeHandler().handle(con_object, con_params, snapshot_policy_details)
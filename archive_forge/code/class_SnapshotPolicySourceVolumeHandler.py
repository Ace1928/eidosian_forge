from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicySourceVolumeHandler:

    def handle(self, con_object, con_params, snapshot_policy_details):
        if snapshot_policy_details and con_params['state'] == 'present' and (con_params['source_volume'] is not None):
            for source_volume_element in range(len(con_params['source_volume'])):
                if not (con_params['source_volume'][source_volume_element]['id'] or con_params['source_volume'][source_volume_element]['name']):
                    con_object.module.fail_json(msg='Either id or name of source volume needs to be passed with state of source volume')
                elif con_params['source_volume'][source_volume_element]['id'] and con_params['source_volume'][source_volume_element]['name']:
                    con_object.module.fail_json(msg='id and name of source volume are mutually exclusive')
                elif con_params['source_volume'][source_volume_element]['id'] or con_params['source_volume'][source_volume_element]['name']:
                    volume_details = con_object.get_volume(vol_id=con_params['source_volume'][source_volume_element]['id'], vol_name=con_params['source_volume'][source_volume_element]['name'])
                    con_object.result['changed'] = con_object.manage_source_volume(snap_pol_details=snapshot_policy_details, vol_details=volume_details, source_volume_element=source_volume_element)
                    snapshot_policy_details = con_object.get_snapshot_policy(snap_pol_name=con_params['snapshot_policy_name'], snap_pol_id=con_params['snapshot_policy_id'])
        SnapshotPolicyPauseHandler().handle(con_object, con_params, snapshot_policy_details)
from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyHandler:

    def handle(self, con_object, con_params):
        access_mode = get_access_mode(con_params['access_mode'])
        snapshot_policy_details = con_object.get_snapshot_policy(snap_pol_name=con_params['snapshot_policy_name'], snap_pol_id=con_params['snapshot_policy_id'])
        auto_snapshot_creation_cadence_in_min = None
        if snapshot_policy_details:
            auto_snapshot_creation_cadence_in_min = snapshot_policy_details['autoSnapshotCreationCadenceInMin']
        msg = f'Fetched the snapshot policy details {str(snapshot_policy_details)}'
        LOG.info(msg)
        if con_params['auto_snapshot_creation_cadence'] is not None:
            auto_snapshot_creation_cadence_in_min = utils.get_time_minutes(time=con_params['auto_snapshot_creation_cadence']['time'], time_unit=con_params['auto_snapshot_creation_cadence']['unit'])
        SnapshotPolicyCreateHandler().handle(con_object, con_params, snapshot_policy_details, access_mode, auto_snapshot_creation_cadence_in_min)
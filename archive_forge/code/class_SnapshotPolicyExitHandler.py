from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
class SnapshotPolicyExitHandler:

    def handle(self, con_object, snapshot_policy_details):
        con_object.result['snapshot_policy_details'] = snapshot_policy_details
        con_object.module.exit_json(**con_object.result)
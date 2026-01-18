from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ReplicationSessionExitHandler:

    def handle(self, session_object, replication_session_obj):
        if replication_session_obj:
            session_object.result['replication_session_details'] = replication_session_obj._get_properties()
        session_object.module.exit_json(**session_object.result)
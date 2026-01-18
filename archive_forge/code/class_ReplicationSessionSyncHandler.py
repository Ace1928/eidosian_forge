from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ReplicationSessionSyncHandler:

    def handle(self, session_object, session_params, replication_session_obj):
        if replication_session_obj and session_params['state'] == 'present' and session_params['sync']:
            session_object.result['changed'] = session_object.sync(replication_session_obj)
            if session_object.result['changed']:
                replication_session_obj = session_object.get_replication_session(session_params['session_id'], session_params['session_name'])
        ReplicationSessionFailoverHandler().handle(session_object, session_params, replication_session_obj)
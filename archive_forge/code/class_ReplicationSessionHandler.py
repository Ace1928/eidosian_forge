from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ReplicationSessionHandler:

    def handle(self, session_object, session_params):
        replication_session_obj = session_object.get_replication_session(session_params['session_id'], session_params['session_name'])
        if session_params['state'] == 'present' and (not replication_session_obj):
            session_object.module.fail_json(msg=f'Replication session {session_params['session_id'] or session_params['session_name']} is invalid.')
        ReplicationSessionPauseHandler().handle(session_object, session_params, replication_session_obj)
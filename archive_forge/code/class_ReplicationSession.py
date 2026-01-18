from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ReplicationSession(object):
    """Class with replication session operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_replication_session_parameters())
        mutually_exclusive = [['session_id', 'session_name']]
        required_one_of = [['session_id', 'session_name']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of)
        utils.ensure_required_libs(self.module)
        self.result = dict(changed=False, replication_session_details={})
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)
        LOG.info('Check Mode Flag %s', self.module.check_mode)

    def get_replication_session(self, id=None, name=None):
        """Get the details of a replication session.
            :param id: The id of the replication session
            :param name: The name of the replication session
            :return: instance of the replication session if exist.
        """
        id_or_name = id if id else name
        errormsg = f'Retrieving details of replication session {id_or_name} failed with error'
        try:
            obj_replication_session = self.unity_conn.get_replication_session(name=name, _id=id)
            LOG.info('Successfully retrieved the replication session object %s ', obj_replication_session)
            if obj_replication_session.existed:
                return obj_replication_session
        except utils.HttpError as e:
            if e.http_status == 401:
                self.module.fail_json(msg=f'Incorrect username or password {str(e)}')
            else:
                msg = f'{errormsg} {str(e)}'
                self.module.fail_json(msg=msg)
        except utils.UnityResourceNotFoundError as e:
            msg = f'{errormsg} {str(e)}'
            LOG.error(msg)
            return None
        except Exception as e:
            msg = f'{errormsg} {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def pause(self, session_obj):
        """Pause the replication session.
            :param session_obj: Replication session object
            :return: True if pause is successful.
        """
        try:
            LOG.info('Pause replication session %s', session_obj.name)
            if session_obj.status.name != utils.ReplicationOpStatusEnum.PAUSED.name:
                if not self.module.check_mode:
                    session_obj.pause()
                return True
        except Exception as e:
            msg = f'Pause replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def resume(self, session_obj, force_full_copy):
        """Resume the replication session.
            :param session_obj: Replication session object
            :param force_full_copy: needed when replication session goes out of sync due to a fault.
            :return: True if resume is successful.
        """
        try:
            LOG.info('Resume replication session %s', session_obj.name)
            if session_obj.status.name in (utils.ReplicationOpStatusEnum.PAUSED.name, utils.ReplicationOpStatusEnum.FAILED_OVER.name, utils.ReplicationOpStatusEnum.FAILED_OVER_WITH_SYNC.name):
                if not self.module.check_mode:
                    session_obj.resume(force_full_copy=force_full_copy)
                return True
        except Exception as e:
            msg = f'Resume replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def sync(self, session_obj):
        """Sync the replication session.
            :param session_obj: Replication session object
            :return: True if sync is successful.
        """
        try:
            LOG.info('Sync replication session %s', session_obj.name)
            if not self.module.check_mode:
                session_obj.sync()
            return True
        except Exception as e:
            msg = f'Sync replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def failover(self, session_obj, sync_failover, force):
        """Failover the replication session.
            :param session_obj: Replication session object
            :param sync_failover: To sync the source and destination resources
            :param force: Skip pre-checks on file system(s) replication sessions of a NAS server
            :return: True if failover is successful.
        """
        try:
            LOG.info('Failover replication session %s', session_obj.name)
            if sync_failover and session_obj.status.name != utils.ReplicationOpStatusEnum.FAILED_OVER_WITH_SYNC.name or (not sync_failover and session_obj.status.name != utils.ReplicationOpStatusEnum.FAILED_OVER.name):
                if not self.module.check_mode:
                    session_obj.failover(sync=sync_failover, force=force)
                return True
        except Exception as e:
            msg = f'Failover replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def failback(self, session_obj, force_full_copy):
        """Failback the replication session.
            :param session_obj: Replication session object
            :param force_full_copy: needed when replication session goes out of sync due to a fault.
            :return: True if failback is successful.
        """
        try:
            LOG.info('Failback replication session %s', session_obj.name)
            if session_obj.status.name in (utils.ReplicationOpStatusEnum.FAILED_OVER.name, utils.ReplicationOpStatusEnum.FAILED_OVER_WITH_SYNC.name, utils.ReplicationOpStatusEnum.PAUSED.name):
                if not self.module.check_mode:
                    session_obj.failback(force_full_copy=force_full_copy)
                return True
        except Exception as e:
            msg = f'Failback replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def delete(self, session_obj):
        """Delete the replication session.
            :param session_obj: Replication session object
            :return: True if delete is successful.
        """
        try:
            LOG.info('Delete replication session %s', session_obj.name)
            if not self.module.check_mode:
                session_obj.delete()
            return True
        except Exception as e:
            msg = f'Deleting replication session {session_obj.name} failed with error {str(e)}'
            LOG.error(msg)
            self.module.fail_json(msg=msg)
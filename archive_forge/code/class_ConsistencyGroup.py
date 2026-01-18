from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class ConsistencyGroup(object):
    """Class with consistency group operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_consistencygroup_parameters())
        mutually_exclusive = [['cg_name', 'cg_id']]
        required_one_of = [['cg_name', 'cg_id']]
        required_together = [['volumes', 'vol_state'], ['hosts', 'mapping_state']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of, required_together=required_together)
        utils.ensure_required_libs(self.module)
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)

    def return_cg_instance(self, cg_name):
        """Return the consistency group instance.
            :param cg_name: The name of the consistency group
            :return: Instance of the consistency group
        """
        try:
            cg_details = self.unity_conn.get_cg(name=cg_name)
            cg_id = cg_details.get_id()
            cg_obj = utils.cg.UnityConsistencyGroup.get(self.unity_conn._cli, cg_id)
            return cg_obj
        except Exception as e:
            msg = 'Failed to get the consistency group {0} instance with error {1}'.format(cg_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_details(self, cg_id=None, cg_name=None):
        """Get consistency group details.
            :param cg_id: The id of the consistency group
            :param cg_name: The name of the consistency group
            :return: Dict containing consistency group details if exists
        """
        id_or_name = cg_id if cg_id else cg_name
        errormsg = 'Failed to get details of consistency group {0} with error {1}'
        try:
            cg_details = self.unity_conn.get_cg(_id=cg_id, name=cg_name)
            if cg_name is None:
                cg_name = cg_details.name
            if cg_details.existed:
                cg_obj = self.return_cg_instance(cg_name)
                snapshots = cg_obj.snapshots
                snapshot_list = [snap._get_properties() for snap in snapshots]
                cg_ret_details = cg_details._get_properties()
                if cg_ret_details['block_host_access']:
                    for i in range(len(cg_details.block_host_access)):
                        cg_ret_details['block_host_access']['UnityBlockHostAccessList'][i]['UnityBlockHostAccess']['id'] = cg_details.block_host_access[i].host.id
                        cg_ret_details['block_host_access']['UnityBlockHostAccessList'][i]['UnityBlockHostAccess']['name'] = cg_details.block_host_access[i].host.name
                cg_ret_details['snapshots'] = snapshot_list
                if cg_ret_details['luns'] is not None:
                    for i in range(len(cg_details.luns)):
                        cg_ret_details['luns']['UnityLunList'][i]['UnityLun']['name'] = cg_details.luns[i].name
                if cg_ret_details['snap_schedule'] is not None:
                    cg_ret_details['snap_schedule']['UnitySnapSchedule']['name'] = cg_details.snap_schedule.name
                cg_ret_details['cg_replication_enabled'] = True if cg_details.check_cg_is_replicated() else False
                return cg_ret_details
            else:
                LOG.info('Failed to get details of consistency group %s', id_or_name)
                return None
        except utils.HttpError as e:
            if e.http_status == 401:
                auth_err = 'Incorrect username or password, {0}'.format(e.message)
                msg = errormsg.format(id_or_name, auth_err)
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            else:
                msg = errormsg.format(id_or_name, str(e))
                LOG.error(msg)
                self.module.fail_json(msg=msg)
        except utils.UnityResourceNotFoundError as e:
            msg = errormsg.format(id_or_name, str(e))
            LOG.error(msg)
            return None
        except Exception as e:
            msg = errormsg.format(id_or_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_host_id_by_name(self, host_name):
        """ Get host ID by host name
        :param host_name: str
        :return: unity host ID
        :rtype: str
        """
        try:
            host_obj = self.unity_conn.get_host(name=host_name)
            if host_obj and host_obj.existed:
                return host_obj.id
            else:
                msg = 'Host name: %s does not exists' % host_name
                LOG.error(msg)
                self.module.fail_json(msg=msg)
        except Exception as e:
            msg = 'Failed to get host ID by name: %s error: %s' % (host_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_volume_details(self, vol_name=None, vol_id=None):
        """Get the details of a volume.
            :param vol_name: The name of the volume
            :param vol_id: The id of the volume
            :return: Dict containing volume details if exists
        """
        id_or_name = vol_id if vol_id else vol_name
        try:
            lun = self.unity_conn.get_lun(name=vol_name, _id=vol_id)
            cg = None
            if lun.existed:
                lunid = lun.get_id()
                unitylun = utils.UnityLun.get(self.unity_conn._cli, lunid)
                if unitylun.cg is not None:
                    cg = unitylun.cg
            else:
                errormsg = 'The volume {0} not found.'.format(id_or_name)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            cg_details = self.get_details(cg_id=self.module.params['cg_id'], cg_name=self.module.params['cg_name'])
            if cg is None:
                return lun._get_properties()['id']
            errormsg = 'The volume {0} is already part of consistency group {1}'.format(id_or_name, cg.name)
            if cg_details is None:
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            if cg.id != cg_details['id']:
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            return lun._get_properties()['id']
        except Exception as e:
            msg = 'Failed to get the volume {0} with error {1}'.format(id_or_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def remove_volumes_from_cg(self, cg_name, volumes):
        """Remove volumes from consistency group.
            :param cg_name: The name of the consistency group
            :param volumes: The list of volumes to be removed
            :return: Boolean value to indicate if volumes are removed from
             consistency group
        """
        cg_details = self.unity_conn.get_cg(name=cg_name)._get_properties()
        existing_volumes_in_cg = cg_details['luns']
        existing_vol_ids = []
        if existing_volumes_in_cg:
            existing_vol_ids = [vol['UnityLun']['id'] for vol in existing_volumes_in_cg['UnityLunList']]
        ids_to_remove = []
        vol_name_list = []
        vol_id_list = []
        for vol in volumes:
            if 'vol_id' in vol and (not vol['vol_id'] in vol_id_list):
                vol_id_list.append(vol['vol_id'])
            elif 'vol_name' in vol and (not vol['vol_name'] in vol_name_list):
                vol_name_list.append(vol['vol_name'])
        'remove volume by name'
        for vol in vol_name_list:
            ids_to_remove.append(self.get_volume_details(vol_name=vol))
        vol_id_list = list(set(vol_id_list + ids_to_remove))
        ids_to_remove = list(set(existing_vol_ids).intersection(set(vol_id_list)))
        LOG.info('Volume IDs to remove %s', ids_to_remove)
        if len(ids_to_remove) == 0:
            return False
        vol_remove_list = []
        for vol in ids_to_remove:
            vol_dict = {'id': vol}
            vol_remove_list.append(vol_dict)
        cg_obj = self.return_cg_instance(cg_name)
        try:
            cg_obj.modify(lun_remove=vol_remove_list)
            return True
        except Exception as e:
            errormsg = 'Remove existing volumes from consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def add_volumes_to_cg(self, cg_name, volumes, tiering_policy):
        """Add volumes to consistency group.
            :param cg_name: The name of the consistency group
            :param volumes: The list of volumes to be added to consistency
             group
            :param tiering_policy: The tiering policy that is to be applied to
            consistency group
            :return: The boolean value to indicate if volumes are added to
             consistency group
        """
        cg_details = self.unity_conn.get_cg(name=cg_name)._get_properties()
        existing_volumes_in_cg = cg_details['luns']
        existing_vol_ids = []
        if existing_volumes_in_cg:
            existing_vol_ids = [vol['UnityLun']['id'] for vol in existing_volumes_in_cg['UnityLunList']]
        ids_to_add = []
        vol_name_list = []
        vol_id_list = []
        all_vol_ids = []
        for vol in volumes:
            if 'vol_id' in vol and (not vol['vol_id'] in vol_id_list):
                vol_id_list.append(vol['vol_id'])
            elif 'vol_name' in vol and (not vol['vol_name'] in vol_name_list):
                vol_name_list.append(vol['vol_name'])
        'add volume by name'
        for vol in vol_name_list:
            ids_to_add.append(self.get_volume_details(vol_name=vol))
        'add volume by id'
        for vol in vol_id_list:
            'verifying if volume id exists in array'
            ids_to_add.append(self.get_volume_details(vol_id=vol))
        all_vol_ids = ids_to_add + existing_vol_ids
        ids_to_add = list(set(all_vol_ids) - set(existing_vol_ids))
        LOG.info('Volume IDs to add %s', ids_to_add)
        if len(ids_to_add) == 0:
            return False
        vol_add_list = []
        for vol in ids_to_add:
            vol_dict = {'id': vol}
            vol_add_list.append(vol_dict)
        cg_obj = self.return_cg_instance(cg_name)
        policy_enum = None
        if tiering_policy:
            if utils.TieringPolicyEnum[tiering_policy]:
                policy_enum = utils.TieringPolicyEnum[tiering_policy]
            else:
                errormsg = 'Invalid choice {0} for tiering policy'.format(tiering_policy)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
        try:
            cg_obj.modify(lun_add=vol_add_list, tiering_policy=policy_enum)
            return True
        except Exception as e:
            errormsg = 'Add existing volumes to consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def map_hosts_to_cg(self, cg_name, add_hosts):
        """Map hosts to consistency group.
            :param cg_name: The name of the consistency group
            :param add_hosts: List of hosts that are to be mapped to cg
            :return: Boolean value to indicate if hosts were mapped to cg
        """
        cg_details = self.unity_conn.get_cg(name=cg_name)
        existing_volumes_in_cg = cg_details.luns
        existing_hosts_in_cg = cg_details.block_host_access
        existing_host_ids = []
        'Get list of existing hosts in consistency group'
        if existing_hosts_in_cg:
            for i in range(len(existing_hosts_in_cg)):
                existing_host_ids.append(existing_hosts_in_cg[i].host.id)
        host_id_list = []
        host_name_list = []
        add_hosts_id = []
        host_add_list = []
        all_hosts = []
        for host in add_hosts:
            if 'host_id' in host and (not host['host_id'] in host_id_list):
                host_id_list.append(host['host_id'])
            elif 'host_name' in host and (not host['host_name'] in host_name_list):
                host_name_list.append(host['host_name'])
        'add hosts by name'
        for host_name in host_name_list:
            add_hosts_id.append(self.get_host_id_by_name(host_name))
        all_hosts = host_id_list + existing_host_ids + add_hosts_id
        add_hosts_id = list(set(all_hosts) - set(existing_host_ids))
        if len(add_hosts_id) == 0:
            return False
        if existing_volumes_in_cg:
            for host_id in add_hosts_id:
                host_dict = {'id': host_id}
                host_add_list.append(host_dict)
            LOG.info('List of hosts to be added to consistency group %s ', host_add_list)
            cg_obj = self.return_cg_instance(cg_name)
            try:
                cg_obj.modify(name=cg_name, host_add=host_add_list)
                return True
            except Exception as e:
                errormsg = 'Adding host to consistency group {0} failed with error {1}'.format(cg_name, str(e))
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)

    def unmap_hosts_to_cg(self, cg_name, remove_hosts):
        """Unmap hosts to consistency group.
            :param cg_name: The name of the consistency group
            :param remove_hosts: List of hosts that are to be unmapped from cg
            :return: Boolean value to indicate if hosts were mapped to cg
        """
        cg_details = self.unity_conn.get_cg(name=cg_name)
        existing_hosts_in_cg = cg_details.block_host_access
        existing_host_ids = []
        'Get host ids existing in consistency group'
        if existing_hosts_in_cg:
            for i in range(len(existing_hosts_in_cg)):
                existing_host_ids.append(existing_hosts_in_cg[i].host.id)
        host_remove_list = []
        host_id_list = []
        host_name_list = []
        remove_hosts_id = []
        for host in remove_hosts:
            if 'host_id' in host and (not host['host_id'] in host_id_list):
                host_id_list.append(host['host_id'])
            elif 'host_name' in host and (not host['host_name'] in host_name_list):
                host_name_list.append(host['host_name'])
        'remove hosts by name'
        for host in host_name_list:
            remove_hosts_id.append(self.get_host_id_by_name(host))
        host_id_list = list(set(host_id_list + remove_hosts_id))
        remove_hosts_id = list(set(existing_host_ids).intersection(set(host_id_list)))
        if len(remove_hosts_id) == 0:
            return False
        for host in remove_hosts_id:
            host_dict = {'id': host}
            host_remove_list.append(host_dict)
        cg_obj = self.return_cg_instance(cg_name)
        try:
            cg_obj.modify(name=cg_name, host_remove=host_remove_list)
            return True
        except Exception as e:
            errormsg = 'Removing host from consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def rename_cg(self, cg_name, new_cg_name):
        """Rename consistency group.
            :param cg_name: The name of the consistency group
            :param new_cg_name: The new name of the consistency group
            :return: Boolean value to indicate if consistency group renamed
        """
        cg_obj = self.return_cg_instance(cg_name)
        try:
            cg_obj.modify(name=new_cg_name)
            return True
        except Exception as e:
            errormsg = 'Rename operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def is_cg_modified(self, cg_details):
        """Check if the desired consistency group state is different from
            existing consistency group.
            :param cg_details: The dict containing consistency group details
            :return: Boolean value to indicate if modification is needed
        """
        modified = False
        if self.module.params['tiering_policy'] and cg_details['luns'] is None and (self.module.params['volumes'] is None):
            self.module.fail_json(msg='The system cannot assign a tiering policy to an empty consistency group.')
        if self.module.params['hosts'] and cg_details['luns'] is None and (self.module.params['volumes'] is None):
            self.module.fail_json(msg='The system cannot assign hosts to an empty consistency group.')
        if (cg_details['description'] is not None and self.module.params['description'] is not None and (cg_details['description'] != self.module.params['description']) or (cg_details['description'] is None and self.module.params['description'] is not None)) or (cg_details['snap_schedule'] is not None and self.module.params['snap_schedule'] is not None and (cg_details['snap_schedule']['UnitySnapSchedule']['name'] != self.module.params['snap_schedule']) or (cg_details['snap_schedule'] is None and self.module.params['snap_schedule'])):
            modified = True
        if cg_details['relocation_policy']:
            tier_policy = cg_details['relocation_policy'].split('.')
            if self.module.params['tiering_policy'] is not None and tier_policy[1] != self.module.params['tiering_policy']:
                modified = True
        return modified

    def create_cg(self, cg_name, description, snap_schedule):
        """Create a consistency group.
            :param cg_name: The name of the consistency group
            :param description: The description of the consistency group
            :param snap_schedule: The name of the snapshot schedule
            :return: The boolean value to indicate if consistency group
             created and also returns the CG object
        """
        try:
            if snap_schedule is not None:
                snap_schedule = {'name': snap_schedule}
            cg_obj = utils.cg.UnityConsistencyGroup.create(self.unity_conn._cli, name=cg_name, description=description, snap_schedule=snap_schedule)
            return (True, cg_obj)
        except Exception as e:
            errormsg = 'Create operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def modify_cg(self, cg_name, description, snap_schedule, tiering_policy):
        """Modify consistency group.
            :param cg_name: The name of the consistency group
            :param description: The description of the consistency group
            :param snap_schedule: The name of the snapshot schedule
            :param tiering_policy: The tiering policy that is to be applied to
            consistency group
            :return: The boolean value to indicate if consistency group
             modified
        """
        cg_obj = self.return_cg_instance(cg_name)
        is_snap_schedule_paused = None
        if self.module.params['snap_schedule'] == '':
            is_snap_schedule_paused = False
        if snap_schedule is not None:
            if snap_schedule == '':
                snap_schedule = {'name': None}
            else:
                snap_schedule = {'name': snap_schedule}
        policy_enum = None
        if tiering_policy:
            if utils.TieringPolicyEnum[tiering_policy]:
                policy_enum = utils.TieringPolicyEnum[tiering_policy]
            else:
                errormsg = 'Invalid choice {0} for tiering policy'.format(tiering_policy)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
        try:
            cg_obj.modify(description=description, snap_schedule=snap_schedule, tiering_policy=policy_enum, is_snap_schedule_paused=is_snap_schedule_paused)
            return True
        except Exception as e:
            errormsg = 'Modify operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def delete_cg(self, cg_name):
        """Delete consistency group.
        :param cg_name: The name of the consistency group
        :return: The boolean value to indicate if consistency group deleted
        """
        cg_obj = self.return_cg_instance(cg_name)
        try:
            cg_obj.delete()
            return True
        except Exception as e:
            errormsg = 'Delete operation of consistency group {0} failed with error {1}'.format(cg_name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def refine_volumes(self, volumes):
        """Refine volumes.
            :param volumes: Volumes that is to be added/removed
            :return: List of volumes with each volume being identified with either
            vol_id or vol_name
        """
        for vol in volumes:
            if vol['vol_id'] is not None and vol['vol_name'] is None:
                del vol['vol_name']
            elif vol['vol_name'] is not None and vol['vol_id'] is None:
                del vol['vol_id']
        return volumes

    def refine_hosts(self, hosts):
        """Refine hosts.
            :param hosts: Hosts that is to be mapped/unmapped
            :return: List of hosts with each host being identified with either
            host_id or host_name
        """
        for host in hosts:
            if host['host_id'] is not None and host['host_name'] is None:
                del host['host_name']
            elif host['host_name'] is not None and host['host_id'] is None:
                del host['host_id']
        return hosts

    def validate_volumes(self, volumes):
        """Validate the volumes.
            :param volumes: List of volumes
        """
        for vol in volumes:
            if 'vol_id' in vol and 'vol_name' in vol:
                errormsg = 'Both name and id are found for volume {0}. No action would be taken. Please specify either name or id.'.format(vol)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'vol_id' in vol and len(vol['vol_id'].strip()) == 0:
                errormsg = 'vol_id is blank. Please specify valid vol_id.'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'vol_name' in vol and len(vol.get('vol_name').strip()) == 0:
                errormsg = 'vol_name is blank. Please specify valid vol_name.'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'vol_name' in vol:
                self.get_volume_details(vol_name=vol['vol_name'])
            elif 'vol_id' in vol:
                self.get_volume_details(vol_id=vol['vol_id'])
            else:
                errormsg = 'Expected either vol_name or vol_id, found neither for volume {0}'.format(vol)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)

    def validate_hosts(self, hosts):
        """Validate hosts.
            :param hosts: List of hosts
        """
        for host in hosts:
            if 'host_id' in host and 'host_name' in host:
                errormsg = 'Both name and id are found for host {0}. No action would be taken. Please specify either name or id.'.format(host)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'host_id' in host and len(host['host_id'].strip()) == 0:
                errormsg = 'host_id is blank. Please specify valid host_id.'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'host_name' in host and len(host.get('host_name').strip()) == 0:
                errormsg = 'host_name is blank. Please specify valid host_name.'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            elif 'host_name' in host:
                self.get_host_id_by_name(host_name=host['host_name'])
            elif 'host_id' in host:
                host_obj = self.unity_conn.get_host(_id=host['host_id'])
                if host_obj is None or host_obj.existed is False:
                    msg = 'Host id: %s does not exists' % host['host_id']
                    LOG.error(msg)
                    self.module.fail_json(msg=msg)
            else:
                errormsg = 'Expected either host_name or host_id, found neither for host {0}'.format(host)
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)

    def update_replication_params(self, replication):
        """ Update replication params """
        if 'replication_type' in replication and replication['replication_type'] == 'remote':
            connection_params = {'unispherehost': replication['remote_system']['remote_system_host'], 'username': replication['remote_system']['remote_system_username'], 'password': replication['remote_system']['remote_system_password'], 'validate_certs': replication['remote_system']['remote_system_verifycert'], 'port': replication['remote_system']['remote_system_port']}
            remote_system_conn = utils.get_unity_unisphere_connection(connection_params, application_type)
            replication['remote_system_name'] = remote_system_conn.name
            if replication['destination_pool_name'] is not None:
                pool_object = remote_system_conn.get_pool(name=replication['destination_pool_name'])
                replication['destination_pool_id'] = pool_object.id
        elif replication['destination_pool_name'] is not None:
            pool_object = self.unity_conn.get_pool(name=replication['destination_pool_name'])
            replication['destination_pool_id'] = pool_object.id

    def get_destination_cg_luns(self, source_lun_list):
        """ Form destination cg lun list """
        destination_cg_lun_list = []
        if source_lun_list is not None:
            for source_lun in source_lun_list:
                destination_cg_lun_info = utils.UnityStorageResource()
                destination_cg_lun_info.name = 'DR_' + source_lun.name
                destination_cg_lun_info.is_thin_enabled = source_lun.is_thin_enabled
                destination_cg_lun_info.size_total = source_lun.size_total
                destination_cg_lun_info.id = source_lun.id
                destination_cg_lun_info.is_data_reduction_enabled = source_lun.is_data_reduction_enabled
                destination_cg_lun_list.append(destination_cg_lun_info)
        return destination_cg_lun_list

    def enable_cg_replication(self, cg_name, replication):
        """ Add replication to the consistency group """
        try:
            self.validate_cg_replication_params(replication)
            cg_object = self.return_cg_instance(cg_name)
            if cg_object.check_cg_is_replicated():
                return False
            self.update_replication_params(replication)
            replication_args_list = {'dst_pool_id': replication['destination_pool_id']}
            if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous':
                replication_args_list['max_time_out_of_sync'] = replication['rpo']
            else:
                replication_args_list['max_time_out_of_sync'] = -1
            if 'replication_type' in replication and replication['replication_type'] == 'remote':
                remote_system_name = replication['remote_system_name']
                remote_system_list = self.unity_conn.get_remote_system()
                for remote_system in remote_system_list:
                    if remote_system.name == remote_system_name:
                        replication_args_list['remote_system'] = remote_system
                        break
                if 'remote_system' not in replication_args_list.keys():
                    errormsg = 'Remote system %s is not found' % remote_system_name
                    LOG.error(errormsg)
                    self.module.fail_json(msg=errormsg)
            source_lun_list = cg_object.luns
            replication_args_list['source_luns'] = self.get_destination_cg_luns(source_lun_list)
            if 'destination_cg_name' in replication and replication['destination_cg_name'] is not None:
                replication_args_list['dst_cg_name'] = replication['destination_cg_name']
            else:
                replication_args_list['dst_cg_name'] = 'DR_' + cg_object.name
            LOG.info(('Enabling replication to the consistency group %s', cg_object.name))
            cg_object.replicate_cg_with_dst_resource_provisioning(**replication_args_list)
            return True
        except Exception as e:
            errormsg = 'Enabling replication to the consistency group %s failed with error %s' % (cg_object.name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def disable_cg_replication(self, cg_name):
        """ Remove replication from the consistency group """
        try:
            cg_object = self.return_cg_instance(cg_name)
            if not cg_object.check_cg_is_replicated():
                return False
            LOG.info(('Disabling replication from the consistency group %s', cg_object.name))
            curr_cg_repl_session = self.unity_conn.get_replication_session(src_resource_id=cg_object.id)
            for repl_session in curr_cg_repl_session:
                repl_session.delete()
            return True
        except Exception as e:
            errormsg = 'Disabling replication to the consistency group %s failed with error %s' % (cg_object.name, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def perform_module_operation(self):
        """
        Perform different actions on consistency group module based on
        parameters chosen in playbook
        """
        cg_name = self.module.params['cg_name']
        cg_id = self.module.params['cg_id']
        description = self.module.params['description']
        volumes = self.module.params['volumes']
        snap_schedule = self.module.params['snap_schedule']
        new_cg_name = self.module.params['new_cg_name']
        tiering_policy = self.module.params['tiering_policy']
        vol_state = self.module.params['vol_state']
        hosts = self.module.params['hosts']
        mapping_state = self.module.params['mapping_state']
        replication = self.module.params['replication_params']
        replication_state = self.module.params['replication_state']
        state = self.module.params['state']
        result = dict(changed=False, create_cg='', modify_cg='', rename_cg='', add_vols_to_cg='', remove_vols_from_cg='', delete_cg='', add_hosts_to_cg='', remove_hosts_from_cg='', consistency_group_details={})
        cg_details = self.get_details(cg_id=cg_id, cg_name=cg_name)
        if cg_name is None and cg_details:
            cg_id = None
            cg_name = cg_details['name']
        if volumes:
            volumes = self.refine_volumes(volumes)
            self.validate_volumes(volumes)
        if hosts:
            hosts = self.refine_hosts(hosts)
            self.validate_hosts(hosts)
        modified = False
        if cg_details:
            modified = self.is_cg_modified(cg_details)
        if vol_state and (not volumes):
            self.module.fail_json(msg='Please specify volumes along with vol_state')
        if mapping_state and (not hosts):
            self.module.fail_json(msg='Please specify hosts along with mapping_state')
        if replication and replication_state is None:
            self.module.fail_json(msg='Please specify replication_state along with replication_params')
        if state == 'present' and (not cg_details):
            if not volumes and tiering_policy:
                self.module.fail_json(msg='The system cannot assign a tiering policy to an empty consistency group')
            if not volumes and hosts:
                self.module.fail_json(msg='The system cannot assign hosts to an empty consistency group')
            if not cg_name:
                msg = 'The parameter cg_name length is 0. It is too short. The min length is 1.'
                self.module.fail_json(msg=msg)
            if new_cg_name:
                self.module.fail_json(msg='Invalid argument, new_cg_name is not required')
            result['create_cg'], cg_details = self.create_cg(cg_name, description, snap_schedule)
        elif state == 'absent' and cg_details:
            if cg_details['cg_replication_enabled']:
                self.module.fail_json(msg='Consistency group cannot be deleted because it is participating in a replication session.')
            if cg_details['luns']:
                self.module.fail_json(msg='Please remove all volumes which are part of consistency group before deleting it.')
            result['delete_cg'] = self.delete_cg(cg_name)
        if state == 'present' and vol_state == 'present-in-group' and cg_details and volumes:
            result['add_vols_to_cg'] = self.add_volumes_to_cg(cg_name, volumes, tiering_policy)
        elif state == 'present' and vol_state == 'absent-in-group' and cg_details and volumes:
            result['remove_vols_from_cg'] = self.remove_volumes_from_cg(cg_name, volumes)
        if hosts and mapping_state == 'mapped' and cg_details:
            result['add_hosts_to_cg'] = self.map_hosts_to_cg(cg_name, hosts)
        if hosts and mapping_state == 'unmapped' and cg_details:
            result['remove_hosts_from_cg'] = self.unmap_hosts_to_cg(cg_name, hosts)
        if state == 'present' and new_cg_name is not None:
            if not new_cg_name:
                msg = 'The parameter new_cg_name length is 0. It is too short. The min length is 1.'
                self.module.fail_json(msg=msg)
            if cg_name != new_cg_name:
                result['rename_cg'] = self.rename_cg(cg_name, new_cg_name)
                cg_name = new_cg_name
        if state == 'present' and cg_details and modified:
            result['modify_cg'] = self.modify_cg(cg_name, description, snap_schedule, tiering_policy)
        if state == 'present' and cg_details and (replication_state is not None):
            if replication_state == 'enable':
                result['changed'] = self.enable_cg_replication(cg_name, replication)
            else:
                result['changed'] = self.disable_cg_replication(cg_name)
        if result['create_cg'] or result['modify_cg'] or result['add_vols_to_cg'] or result['remove_vols_from_cg'] or result['delete_cg'] or result['rename_cg'] or result['add_hosts_to_cg'] or result['remove_hosts_from_cg']:
            result['changed'] = True
        result['consistency_group_details'] = self.get_details(cg_id=cg_id, cg_name=cg_name)
        self.module.exit_json(**result)

    def validate_destination_pool_info(self, replication):
        if replication['destination_pool_id'] is not None and replication['destination_pool_name'] is not None:
            errormsg = "'destination_pool_id' and 'destination_pool_name' is mutually exclusive."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if replication['destination_pool_id'] is None and replication['destination_pool_name'] is None:
            errormsg = "Either 'destination_pool_id' or 'destination_pool_name' is required."
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def validate_replication_mode(self, replication):
        if 'replication_mode' in replication and replication['replication_mode'] == 'asynchronous':
            if replication['rpo'] is None:
                errormsg = "rpo is required together with 'asynchronous' replication_mode."
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            if replication['rpo'] < 5 or replication['rpo'] > 1440:
                errormsg = 'rpo value should be in range of 5 to 1440'
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)

    def validate_cg_replication_params(self, replication):
        """ Validate cg replication params """
        if replication is None:
            errormsg = 'Please specify replication_params to enable replication.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        else:
            self.validate_destination_pool_info(replication)
            self.validate_replication_mode(replication)
            if replication['replication_type'] == 'remote' and replication['remote_system'] is None:
                errormsg = "remote_system is required together with 'remote' replication_type"
                LOG.error(errormsg)
                self.module.fail_json(msg=errormsg)
            if 'destination_cg_name' in replication and replication['destination_cg_name'] is not None:
                dst_cg_name_length = len(replication['destination_cg_name'])
                if dst_cg_name_length == 0 or dst_cg_name_length > 95:
                    errormsg = 'destination_cg_name value should be in range of 1 to 95'
                    LOG.error(errormsg)
                    self.module.fail_json(msg=errormsg)
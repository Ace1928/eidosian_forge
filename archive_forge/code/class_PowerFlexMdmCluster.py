from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
class PowerFlexMdmCluster(object):
    """Class with MDM cluster operations"""

    def __init__(self):
        """ Define all parameters required by this module"""
        self.module_params = utils.get_powerflex_gateway_host_parameters()
        self.module_params.update(get_powerflex_mdm_cluster_parameters())
        mut_ex_args = [['mdm_name', 'mdm_id'], ['virtual_ip_interfaces', 'clear_interfaces']]
        required_together_args = [['cluster_mode', 'mdm', 'mdm_state']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True, mutually_exclusive=mut_ex_args, required_together=required_together_args)
        utils.ensure_required_libs(self.module)
        self.not_exist_msg = 'MDM {0} does not exists in MDM cluster.'
        self.exist_msg = 'MDM already exists in the MDM cluster'
        try:
            self.powerflex_conn = utils.get_powerflex_gateway_host_connection(self.module.params)
            LOG.info('Got the PowerFlex system connection object instance')
            LOG.info('Check Mode Flag %s', self.module.check_mode)
        except Exception as e:
            LOG.error(str(e))
            self.module.fail_json(msg=str(e))

    def set_mdm_virtual_interface(self, mdm_id=None, mdm_name=None, virtual_ip_interfaces=None, clear_interfaces=None, mdm_cluster_details=None):
        """Modify the MDM virtual IP interface.
        :param mdm_id: ID of MDM
        :param mdm_name: Name of MDM
        :param virtual_ip_interfaces: List of virtual IP interfaces
        :param clear_interfaces: clear virtual IP interfaces of MDM.
        :param mdm_cluster_details: Details of MDM cluster
        :return: True if modification of virtual interface or clear operation
                 successful
        """
        name_or_id = mdm_id if mdm_id else mdm_name
        if mdm_name is None and mdm_id is None:
            err_msg = 'Please provide mdm_name/mdm_id to modify virtual IP interfaces the MDM.'
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
        mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=mdm_cluster_details)
        if mdm_details is None:
            err_msg = self.not_exist_msg.format(name_or_id)
            self.module.fail_json(msg=err_msg)
        mdm_id = mdm_details['id']
        modify_list = []
        modify_list, clear = is_modify_mdm_virtual_interface(virtual_ip_interfaces, clear_interfaces, mdm_details)
        if modify_list is None and (not clear):
            LOG.info('No change required in MDM virtual IP interfaces.')
            return False
        try:
            log_msg = 'Modifying MDM virtual interfaces to %s or %s' % (str(modify_list), clear)
            LOG.debug(log_msg)
            if not self.module.check_mode:
                self.powerflex_conn.system.modify_virtual_ip_interface(mdm_id=mdm_id, virtual_ip_interfaces=modify_list, clear_interfaces=clear)
            return True
        except Exception as e:
            error_msg = 'Failed to modify the virtual IP interfaces of MDM {0} with error {1}'.format(name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def set_performance_profile(self, performance_profile=None, cluster_details=None):
        """ Set the performance profile of Cluster MDMs
        :param performance_profile: Specifies the performance profile of MDMs
        :param cluster_details: Details of MDM cluster
        :return: True if updated successfully
        """
        if self.module.params['state'] == 'present' and performance_profile:
            if cluster_details['perfProfile'] != performance_profile:
                try:
                    if not self.module.check_mode:
                        self.powerflex_conn.system.set_cluster_mdm_performance_profile(performance_profile=performance_profile)
                    return True
                except Exception as e:
                    error_msg = 'Failed to update performance profile to {0} with error {1}.'.format(performance_profile, str(e))
                    LOG.error(error_msg)
                    self.module.fail_json(msg=error_msg)
            return False
        return False

    def rename_mdm(self, mdm_name=None, mdm_id=None, mdm_new_name=None, cluster_details=None):
        """Rename the MDM
        :param mdm_name: Name of the MDM.
        :param mdm_id: ID of the MDM.
        :param mdm_new_name: New name of the MDM.
        :param cluster_details: Details of the MDM cluster.
        :return: True if successfully renamed.
        """
        name_or_id = mdm_id if mdm_id else mdm_name
        if mdm_name is None and mdm_id is None:
            err_msg = 'Please provide mdm_name/mdm_id to rename the MDM.'
            self.module.fail_json(msg=err_msg)
        mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
        if mdm_details is None:
            err_msg = self.not_exist_msg.format(name_or_id)
            self.module.fail_json(msg=err_msg)
        mdm_id = mdm_details['id']
        try:
            if 'name' in mdm_details and mdm_new_name != mdm_details['name'] or 'name' not in mdm_details:
                log_msg = 'Modifying the MDM name from %s to %s.' % (mdm_name, mdm_new_name)
                LOG.info(log_msg)
                if not self.module.check_mode:
                    self.powerflex_conn.system.rename_mdm(mdm_id=mdm_id, mdm_new_name=mdm_new_name)
                return True
        except Exception as e:
            error_msg = 'Failed to rename the MDM {0} with error {1}.'.format(name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def is_none_name_id_in_switch_cluster_mode(self, mdm):
        """ Check whether mdm dict have mdm_name and mdm_id or not"""
        for node in mdm:
            if node['mdm_id'] and node['mdm_name']:
                msg = 'parameters are mutually exclusive: mdm_name|mdm_id'
                self.module.fail_json(msg=msg)

    def change_cluster_mode(self, cluster_mode, mdm, cluster_details):
        """change the MDM cluster mode.
        :param cluster_mode: specifies the mode of MDM cluster
        :param mdm: A dict containing parameters to change MDM cluster mode
        :param cluster_details: Details of MDM cluster
        :return: True if mode changed successfully
        """
        self.is_none_name_id_in_switch_cluster_mode(mdm=mdm)
        if cluster_mode == cluster_details['clusterMode']:
            LOG.info('MDM cluster is already in required mode.')
            return False
        add_secondary = []
        add_tb = []
        remove_secondary = []
        remove_tb = []
        if self.module.params['state'] == 'present' and self.module.params['mdm_state'] == 'present-in-cluster':
            add_secondary, add_tb = self.cluster_expand_list(mdm, cluster_details)
        elif self.module.params['state'] == 'present' and self.module.params['mdm_state'] == 'absent-in-cluster':
            remove_secondary, remove_tb = self.cluster_reduce_list(mdm, cluster_details)
        try:
            if not self.module.check_mode:
                self.powerflex_conn.system.switch_cluster_mode(cluster_mode=cluster_mode, add_secondary=add_secondary, remove_secondary=remove_secondary, add_tb=add_tb, remove_tb=remove_tb)
            return True
        except Exception as e:
            err_msg = 'Failed to change the MDM cluster mode with error {0}'.format(str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)

    def gather_secondarys_ids(self, mdm, cluster_details):
        """ Prepare a list of secondary MDMs for switch cluster mode
            operation"""
        secondarys = []
        for node in mdm:
            name_or_id = node['mdm_name'] if node['mdm_name'] else node['mdm_id']
            if node['mdm_type'] == 'Secondary' and node['mdm_id'] is not None:
                mdm_details = self.is_mdm_name_id_exists(mdm_id=node['mdm_id'], cluster_details=cluster_details)
                if mdm_details is None:
                    err_msg = self.not_exist_msg.format(name_or_id)
                    self.module.fail_json(msg=err_msg)
                secondarys.append(node['mdm_id'])
            elif node['mdm_type'] == 'Secondary' and node['mdm_name'] is not None:
                mdm_details = self.is_mdm_name_id_exists(mdm_name=node['mdm_name'], cluster_details=cluster_details)
                if mdm_details is None:
                    err_msg = self.not_exist_msg.format(name_or_id)
                    self.module.fail_json(msg=err_msg)
                else:
                    secondarys.append(mdm_details['id'])
        return secondarys

    def cluster_expand_list(self, mdm, cluster_details):
        """Whether MDM cluster expansion is required or not.
        """
        add_secondary = []
        add_tb = []
        if 'standbyMDMs' not in cluster_details:
            err_msg = 'No Standby MDMs found. To expand cluster size, first add standby MDMs.'
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
        add_secondary = self.gather_secondarys_ids(mdm, cluster_details)
        for node in mdm:
            name_or_id = node['mdm_name'] if node['mdm_name'] else node['mdm_id']
            if node['mdm_type'] == 'TieBreaker' and node['mdm_id'] is not None:
                add_tb.append(node['mdm_id'])
            elif node['mdm_type'] == 'TieBreaker' and node['mdm_name'] is not None:
                mdm_details = self.is_mdm_name_id_exists(mdm_name=node['mdm_name'], cluster_details=cluster_details)
                if mdm_details is None:
                    err_msg = self.not_exist_msg.format(name_or_id)
                    self.module.fail_json(msg=err_msg)
                else:
                    add_tb.append(mdm_details['id'])
        log_msg = 'expand List are: %s, %s' % (add_secondary, add_tb)
        LOG.debug(log_msg)
        return (add_secondary, add_tb)

    def cluster_reduce_list(self, mdm, cluster_details):
        """Whether MDM cluster reduction is required or not.
        """
        remove_secondary = []
        remove_tb = []
        remove_secondary = self.gather_secondarys_ids(mdm, cluster_details)
        for node in mdm:
            name_or_id = node['mdm_name'] if node['mdm_name'] else node['mdm_id']
            if node['mdm_type'] == 'TieBreaker' and node['mdm_id'] is not None:
                mdm_details = self.is_mdm_name_id_exists(mdm_id=node['mdm_id'], cluster_details=cluster_details)
                if mdm_details is None:
                    err_msg = self.not_exist_msg.format(name_or_id)
                    self.module.fail_json(msg=err_msg)
                remove_tb.append(mdm_details['id'])
            elif node['mdm_type'] == 'TieBreaker' and node['mdm_name'] is not None:
                mdm_details = self.is_mdm_name_id_exists(mdm_name=node['mdm_name'], cluster_details=cluster_details)
                if mdm_details is None:
                    err_msg = self.not_exist_msg.format(name_or_id)
                    self.module.fail_json(msg=err_msg)
                else:
                    remove_tb.append(mdm_details['id'])
        log_msg = 'Reduce List are: %s, %s.' % (remove_secondary, remove_tb)
        LOG.debug(log_msg)
        return (remove_secondary, remove_tb)

    def perform_add_standby(self, mdm_name, standby_payload):
        """ Perform SDK call to add a standby MDM

        :param mdm_name: Name of new standby MDM
        :param standby_payload: Parameters dict to add a standby MDM
        :return: True if standby MDM added successfully
        """
        try:
            if not self.module.check_mode:
                self.powerflex_conn.system.add_standby_mdm(mdm_ips=standby_payload['mdm_ips'], role=standby_payload['role'], management_ips=standby_payload['management_ips'], mdm_name=mdm_name, port=standby_payload['port'], allow_multiple_ips=standby_payload['allow_multiple_ips'], virtual_interface=standby_payload['virtual_interfaces'])
            return True
        except Exception as e:
            err_msg = 'Failed to Add a standby MDM with error {0}.'.format(str(e))
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)

    def is_id_new_name_in_add_mdm(self):
        """ Check whether mdm_id or mdm_new_name present in Add standby MDM"""
        if self.module.params['mdm_id'] or self.module.params['mdm_new_name']:
            err_msg = 'Parameters mdm_id/mdm_new_name are not allowed while adding a standby MDM. Please try with valid parameters to add a standby MDM.'
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)

    def add_standby_mdm(self, mdm_name, standby_mdm, cluster_details):
        """ Adding a standby MDM"""
        if self.module.params['state'] == 'present' and standby_mdm is not None and self.check_mdm_exists(standby_mdm['mdm_ips'], cluster_details):
            self.is_id_new_name_in_add_mdm()
            mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, cluster_details=cluster_details)
            if mdm_details:
                LOG.info('Standby MDM %s exits in the system', mdm_name)
                return (False, cluster_details)
            standby_payload = prepare_standby_payload(standby_mdm)
            standby_add = self.perform_add_standby(mdm_name, standby_payload)
            if standby_add:
                cluster_details = self.get_mdm_cluster_details()
                msg = 'Fetched the MDM cluster details {0} after adding a standby MDM'.format(str(cluster_details))
                LOG.info(msg)
                return (True, cluster_details)
        return (False, cluster_details)

    def remove_standby_mdm(self, mdm_name, mdm_id, cluster_details):
        """ Remove the Standby MDM
        :param mdm_id: ID of MDM that will become owner of MDM cluster
        :param mdm_name: Name of MDM that will become owner of MDM cluster
        :param cluster_details: Details of MDM cluster
        :return: True if MDM removed successful
        """
        name_or_id = mdm_id if mdm_id else mdm_name
        if mdm_id is None and mdm_name is None:
            err_msg = 'Either mdm_name or mdm_id is required while removing the standby MDM.'
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
        mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
        if mdm_details is None:
            LOG.info('MDM %s not exists in MDM cluster.', name_or_id)
            return False
        mdm_id = mdm_details['id']
        try:
            if not self.module.check_mode:
                self.powerflex_conn.system.remove_standby_mdm(mdm_id=mdm_id)
            return True
        except Exception as e:
            error_msg = 'Failed to remove the standby MDM {0} from the MDM cluster with error {1}'.format(name_or_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def change_ownership(self, mdm_id=None, mdm_name=None, cluster_details=None):
        """ Change the ownership of MDM cluster.
        :param mdm_id: ID of MDM that will become owner of MDM cluster
        :param mdm_name: Name of MDM that will become owner of MDM cluster
        :param cluster_details: Details of MDM cluster
        :return: True if Owner of MDM cluster change successful
        """
        name_or_id = mdm_id if mdm_id else mdm_name
        if mdm_id is None and mdm_name is None:
            err_msg = 'Either mdm_name or mdm_id is required while changing ownership of MDM cluster.'
            LOG.error(err_msg)
            self.module.fail_json(msg=err_msg)
        mdm_details = self.is_mdm_name_id_exists(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details)
        if mdm_details is None:
            err_msg = self.not_exist_msg.format(name_or_id)
            self.module.fail_json(msg=err_msg)
        mdm_id = mdm_details['id']
        if mdm_details['id'] == cluster_details['master']['id']:
            LOG.info('MDM %s is already Owner of MDM cluster.', name_or_id)
            return False
        else:
            try:
                if not self.module.check_mode:
                    self.powerflex_conn.system.change_mdm_ownership(mdm_id=mdm_id)
                return True
            except Exception as e:
                error_msg = 'Failed to update the Owner of MDM cluster to MDM {0} with error {1}'.format(name_or_id, str(e))
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)

    def find_mdm_in_secondarys(self, mdm_name=None, mdm_id=None, cluster_details=None, name_or_id=None):
        """Whether MDM exists with mdm_name or id in secondary MDMs"""
        if 'slaves' in cluster_details:
            for mdm in cluster_details['slaves']:
                if 'name' in mdm and mdm_name == mdm['name'] or mdm_id == mdm['id']:
                    LOG.info('MDM %s found in Secondarys MDM.', name_or_id)
                    return mdm

    def find_mdm_in_tb(self, mdm_name=None, mdm_id=None, cluster_details=None, name_or_id=None):
        """Whether MDM exists with mdm_name or id in tie-breaker MDMs"""
        if 'tieBreakers' in cluster_details:
            for mdm in cluster_details['tieBreakers']:
                if 'name' in mdm and mdm_name == mdm['name'] or mdm_id == mdm['id']:
                    LOG.info('MDM %s found in tieBreakers MDM.', name_or_id)
                    return mdm

    def find_mdm_in_standby(self, mdm_name=None, mdm_id=None, cluster_details=None, name_or_id=None):
        """Whether MDM exists with mdm_name or id in standby MDMs"""
        if 'standbyMDMs' in cluster_details:
            for mdm in cluster_details['standbyMDMs']:
                if 'name' in mdm and mdm_name == mdm['name'] or mdm_id == mdm['id']:
                    LOG.info('MDM %s found in standby MDM.', name_or_id)
                    return mdm

    def is_mdm_name_id_exists(self, mdm_id=None, mdm_name=None, cluster_details=None):
        """Whether MDM exists with mdm_name or id """
        name_or_id = mdm_id if mdm_id else mdm_name
        if 'name' in cluster_details['master'] and mdm_name == cluster_details['master']['name'] or mdm_id == cluster_details['master']['id']:
            LOG.info('MDM %s is master MDM.', name_or_id)
            return cluster_details['master']
        secondary_mdm = []
        secondary_mdm = self.find_mdm_in_secondarys(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
        if secondary_mdm is not None:
            return secondary_mdm
        tb_mdm = []
        tb_mdm = self.find_mdm_in_tb(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
        if tb_mdm is not None:
            return tb_mdm
        standby_mdm = self.find_mdm_in_standby(mdm_name=mdm_name, mdm_id=mdm_id, cluster_details=cluster_details, name_or_id=name_or_id)
        if standby_mdm is not None:
            return standby_mdm
        LOG.info('MDM %s does not exists in MDM Cluster.', name_or_id)
        return None

    def get_mdm_cluster_details(self):
        """Get MDM cluster details
        :return: Details of MDM Cluster if existed.
        """
        try:
            mdm_cluster_details = self.powerflex_conn.system.get_mdm_cluster_details()
            if len(mdm_cluster_details) == 0:
                msg = 'MDM cluster not found'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            resp = self.get_system_details()
            if resp is not None:
                mdm_cluster_details['perfProfile'] = resp['perfProfile']
            gateway_configuration_details = self.powerflex_conn.system.get_gateway_configuration_details()
            if gateway_configuration_details is not None:
                mdm_cluster_details['mdmAddresses'] = gateway_configuration_details['mdmAddresses']
            return mdm_cluster_details
        except Exception as e:
            error_msg = 'Failed to get the MDM cluster with error {0}.'
            error_msg = error_msg.format(str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def check_ip_in_secondarys(self, standby_ip, cluster_details):
        """whether standby IPs present in secondary MDMs"""
        if 'slaves' in cluster_details:
            for secondary_mdm in cluster_details['slaves']:
                current_secondary_ips = secondary_mdm['ips']
                for ips in standby_ip:
                    if ips in current_secondary_ips:
                        LOG.info(self.exist_msg)
                        return False
        return True

    def check_ip_in_tbs(self, standby_ip, cluster_details):
        """whether standby IPs present in tie-breaker MDMs"""
        if 'tieBreakers' in cluster_details:
            for tb_mdm in cluster_details['tieBreakers']:
                current_tb_ips = tb_mdm['ips']
                for ips in standby_ip:
                    if ips in current_tb_ips:
                        LOG.info(self.exist_msg)
                        return False
        return True

    def check_ip_in_standby(self, standby_ip, cluster_details):
        """whether standby IPs present in standby MDMs"""
        if 'standbyMDMs' in cluster_details:
            for stb_mdm in cluster_details['standbyMDMs']:
                current_stb_ips = stb_mdm['ips']
                for ips in standby_ip:
                    if ips in current_stb_ips:
                        LOG.info(self.exist_msg)
                        return False
        return True

    def check_mdm_exists(self, standby_ip=None, cluster_details=None):
        """Check whether standby MDM exists in MDM Cluster"""
        current_master_ips = cluster_details['master']['ips']
        for ips in standby_ip:
            if ips in current_master_ips:
                LOG.info(self.exist_msg)
                return False
        in_secondary = self.check_ip_in_secondarys(standby_ip=standby_ip, cluster_details=cluster_details)
        if not in_secondary:
            return False
        in_tbs = self.check_ip_in_tbs(standby_ip=standby_ip, cluster_details=cluster_details)
        if not in_tbs:
            return False
        in_standby = self.check_ip_in_standby(standby_ip=standby_ip, cluster_details=cluster_details)
        if not in_standby:
            return False
        LOG.info('New Standby MDM does not exists in MDM cluster')
        return True

    def get_system_details(self):
        """Get system details
        :return: Details of PowerFlex system
        """
        try:
            resp = self.powerflex_conn.system.get()
            if len(resp) == 0:
                self.module.fail_json(msg='No system exist on the given host.')
            if len(resp) > 1:
                self.module.fail_json(msg='Multiple systems exist on the given host.')
            return resp[0]
        except Exception as e:
            msg = 'Failed to get system id with error %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def validate_parameters(self):
        """Validate the input parameters"""
        name_params = ['mdm_name', 'mdm_id', 'mdm_new_name']
        msg = 'Please provide the valid {0}'
        for n_item in name_params:
            if self.module.params[n_item] is not None and (len(self.module.params[n_item].strip()) or self.module.params[n_item].count(' ') > 0) == 0:
                err_msg = msg.format(n_item)
                self.module.fail_json(msg=err_msg)

    def perform_module_operation(self):
        """
        Perform different actions on MDM cluster based on parameters passed in
        the playbook
        """
        mdm_name = self.module.params['mdm_name']
        mdm_id = self.module.params['mdm_id']
        mdm_new_name = self.module.params['mdm_new_name']
        standby_mdm = copy.deepcopy(self.module.params['standby_mdm'])
        is_primary = self.module.params['is_primary']
        cluster_mode = self.module.params['cluster_mode']
        mdm = copy.deepcopy(self.module.params['mdm'])
        mdm_state = self.module.params['mdm_state']
        virtual_ip_interfaces = self.module.params['virtual_ip_interfaces']
        clear_interfaces = self.module.params['clear_interfaces']
        performance_profile = self.module.params['performance_profile']
        state = self.module.params['state']
        changed = False
        result = dict(changed=False, mdm_cluster_details={})
        self.validate_parameters()
        mdm_cluster_details = self.get_mdm_cluster_details()
        msg = 'Fetched the MDM cluster details {0}'.format(str(mdm_cluster_details))
        LOG.info(msg)
        standby_changed = False
        performance_changed = False
        renamed_changed = False
        interface_changed = False
        remove_changed = False
        mode_changed = False
        owner_changed = False
        standby_changed, mdm_cluster_details = self.add_standby_mdm(mdm_name, standby_mdm, mdm_cluster_details)
        performance_changed = self.set_performance_profile(performance_profile, mdm_cluster_details)
        if state == 'present' and mdm_new_name:
            renamed_changed = self.rename_mdm(mdm_name, mdm_id, mdm_new_name, mdm_cluster_details)
        if state == 'present' and (virtual_ip_interfaces or clear_interfaces):
            interface_changed = self.set_mdm_virtual_interface(mdm_id, mdm_name, virtual_ip_interfaces, clear_interfaces, mdm_cluster_details)
        if state == 'present' and cluster_mode and mdm and mdm_state:
            mode_changed = self.change_cluster_mode(cluster_mode, mdm, mdm_cluster_details)
        if state == 'absent':
            remove_changed = self.remove_standby_mdm(mdm_name, mdm_id, mdm_cluster_details)
        if state == 'present' and is_primary:
            owner_changed = self.change_ownership(mdm_id, mdm_name, mdm_cluster_details)
        changed = update_change_flag(standby_changed, performance_changed, renamed_changed, interface_changed, mode_changed, remove_changed, owner_changed)
        if owner_changed:
            mdm_cluster_details = {}
        else:
            mdm_cluster_details = self.get_mdm_cluster_details()
        result['mdm_cluster_details'] = mdm_cluster_details
        result['changed'] = changed
        self.module.exit_json(**result)
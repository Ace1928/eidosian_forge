from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell import utils
class CIFSServer(object):
    """Class with CIFS server operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_cifs_server_parameters())
        mutually_exclusive = [['nas_server_name', 'nas_server_id'], ['cifs_server_id', 'cifs_server_name'], ['cifs_server_id', 'netbios_name']]
        required_one_of = [['cifs_server_id', 'cifs_server_name', 'netbios_name', 'nas_server_name', 'nas_server_id']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=True, mutually_exclusive=mutually_exclusive, required_one_of=required_one_of)
        utils.ensure_required_libs(self.module)
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)
        LOG.info('Check Mode Flag %s', self.module.check_mode)

    def get_details(self, cifs_server_id=None, cifs_server_name=None, netbios_name=None, nas_server_id=None):
        """Get CIFS server details.
            :param: cifs_server_id: The ID of the CIFS server
            :param: cifs_server_name: The name of the CIFS server
            :param: netbios_name: Name of the SMB server in windows network
            :param: nas_server_id: The ID of the NAS server
            :return: Dict containing CIFS server details if exists
        """
        LOG.info('Getting CIFS server details')
        id_or_name = get_id_name(cifs_server_id, cifs_server_name, netbios_name, nas_server_id)
        try:
            if cifs_server_id:
                cifs_server_details = self.unity_conn.get_cifs_server(_id=cifs_server_id)
                return process_response(cifs_server_details)
            if cifs_server_name:
                cifs_server_details = self.unity_conn.get_cifs_server(name=cifs_server_name)
                return process_response(cifs_server_details)
            if netbios_name:
                cifs_server_details = self.unity_conn.get_cifs_server(netbios_name=netbios_name)
                if len(cifs_server_details) > 0:
                    return process_dict(cifs_server_details._get_properties())
            if nas_server_id:
                cifs_server_details = self.unity_conn.get_cifs_server(nas_server=nas_server_id)
                if len(cifs_server_details) > 0:
                    return process_dict(cifs_server_details._get_properties())
            return None
        except utils.HttpError as e:
            if e.http_status == 401:
                msg = 'Failed to get CIFS server: %s due to incorrect username/password error: %s' % (id_or_name, str(e))
            else:
                msg = 'Failed to get CIFS server: %s with error: %s' % (id_or_name, str(e))
        except utils.UnityResourceNotFoundError:
            msg = 'CIFS server with ID %s not found' % cifs_server_id
            LOG.info(msg)
            return None
        except utils.StoropsConnectTimeoutError as e:
            msg = 'Failed to get CIFS server: %s with error: %s. Please check unispherehost IP: %s' % (id_or_name, str(e), self.module.params['unispherehost'])
        except Exception as e:
            msg = 'Failed to get details of CIFS server: %s with error: %s' % (id_or_name, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)

    def get_cifs_server_instance(self, cifs_server_id):
        """Get CIFS server instance.
            :param: cifs_server_id: The ID of the CIFS server
            :return: Return CIFS server instance if exists
        """
        try:
            cifs_server_obj = utils.UnityCifsServer.get(cli=self.unity_conn._cli, _id=cifs_server_id)
            return cifs_server_obj
        except Exception as e:
            error_msg = 'Failed to get the CIFS server %s instance with error %s' % (cifs_server_id, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def delete_cifs_server(self, cifs_server_id, skip_unjoin=None, domain_username=None, domain_password=None):
        """Delete CIFS server.
            :param: cifs_server_id: The ID of the CIFS server
            :param: skip_unjoin: Flag indicating whether to unjoin SMB server account from AD before deletion
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :return: Return True if CIFS server is deleted
        """
        LOG.info('Deleting CIFS server')
        try:
            if not self.module.check_mode:
                cifs_obj = self.get_cifs_server_instance(cifs_server_id=cifs_server_id)
                cifs_obj.delete(skip_domain_unjoin=skip_unjoin, username=domain_username, password=domain_password)
            return True
        except Exception as e:
            msg = 'Failed to delete CIFS server: %s with error: %s' % (cifs_server_id, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def get_nas_server_id(self, nas_server_name):
        """Get NAS server ID.
            :param: nas_server_name: The name of NAS server
            :return: Return NAS server ID if exists
        """
        LOG.info('Getting NAS server ID')
        try:
            obj_nas = self.unity_conn.get_nas_server(name=nas_server_name)
            return obj_nas.get_id()
        except Exception as e:
            msg = 'Failed to get details of NAS server: %s with error: %s' % (nas_server_name, str(e))
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def is_modify_interfaces(self, cifs_server_details):
        """Check if modification is required in existing interfaces
            :param: cifs_server_details: CIFS server details
            :return: Flag indicating if modification is required
        """
        existing_interfaces = []
        if cifs_server_details['file_interfaces']['UnityFileInterfaceList']:
            for interface in cifs_server_details['file_interfaces']['UnityFileInterfaceList']:
                existing_interfaces.append(interface['UnityFileInterface']['id'])
        for interface in self.module.params['interfaces']:
            if interface not in existing_interfaces:
                return True
        return False

    def is_modification_required(self, cifs_server_details):
        """Check if modification is required in existing CIFS server
            :param: cifs_server_details: CIFS server details
            :return: Flag indicating if modification is required
        """
        LOG.info('Checking if any modification is required')
        param_list = ['netbios_name', 'workgroup']
        for param in param_list:
            if self.module.params[param] is not None and cifs_server_details[param] is not None and (self.module.params[param].upper() != cifs_server_details[param]):
                return True
        if self.module.params['domain'] is not None and cifs_server_details['domain'] is not None and (self.module.params['domain'] != cifs_server_details['domain']):
            return True
        if self.module.params['interfaces'] is not None:
            return self.is_modify_interfaces(cifs_server_details)
        return False

    def create_cifs_server(self, nas_server_id, interfaces=None, netbios_name=None, cifs_server_name=None, domain=None, domain_username=None, domain_password=None, workgroup=None, local_password=None):
        """Create CIFS server.
            :param: nas_server_id: The ID of NAS server
            :param: interfaces: List of file interfaces
            :param: netbios_name: Name of the SMB server in windows network
            :param: cifs_server_name: Name of the CIFS server
            :param: domain: The domain name where the SMB server is registered in Active Directory
            :param: domain_username: The domain username
            :param: domain_password: The domain password
            :param: workgroup: Standalone SMB server workgroup
            :param: local_password: Standalone SMB server admin password
            :return: Return True if CIFS server is created
        """
        LOG.info('Creating CIFS server')
        try:
            if not self.module.check_mode:
                utils.UnityCifsServer.create(cli=self.unity_conn._cli, nas_server=nas_server_id, interfaces=interfaces, netbios_name=netbios_name, name=cifs_server_name, domain=domain, domain_username=domain_username, domain_password=domain_password, workgroup=workgroup, local_password=local_password)
            return True
        except Exception as e:
            msg = 'Failed to create CIFS server with error: %s' % str(e)
            LOG.error(msg)
            self.module.fail_json(msg=msg)

    def validate_params(self):
        """Validate the parameters
        """
        param_list = ['nas_server_id', 'nas_server_name', 'domain', 'cifs_server_id', 'cifs_server_name', 'local_password', 'netbios_name', 'workgroup', 'domain_username', 'domain_password']
        msg = 'Please provide valid {0}'
        for param in param_list:
            if self.module.params[param] is not None and len(self.module.params[param].strip()) == 0:
                errmsg = msg.format(param)
                self.module.fail_json(msg=errmsg)

    def perform_module_operation(self):
        """
        Perform different actions on CIFS server module based on parameters
        passed in the playbook
        """
        cifs_server_id = self.module.params['cifs_server_id']
        cifs_server_name = self.module.params['cifs_server_name']
        nas_server_id = self.module.params['nas_server_id']
        nas_server_name = self.module.params['nas_server_name']
        netbios_name = self.module.params['netbios_name']
        workgroup = self.module.params['workgroup']
        local_password = self.module.params['local_password']
        domain = self.module.params['domain']
        domain_username = self.module.params['domain_username']
        domain_password = self.module.params['domain_password']
        interfaces = self.module.params['interfaces']
        unjoin_cifs_server_account = self.module.params['unjoin_cifs_server_account']
        state = self.module.params['state']
        result = dict(changed=False, cifs_server_details={})
        self.validate_params()
        if nas_server_name is not None:
            nas_server_id = self.get_nas_server_id(nas_server_name)
        cifs_server_details = self.get_details(cifs_server_id=cifs_server_id, cifs_server_name=cifs_server_name, netbios_name=netbios_name, nas_server_id=nas_server_id)
        if cifs_server_details:
            if cifs_server_id is None:
                cifs_server_id = cifs_server_details['id']
            modify_flag = self.is_modification_required(cifs_server_details)
            if modify_flag:
                self.module.fail_json(msg='Modification is not supported through Ansible module')
        if not cifs_server_details and state == 'present':
            if not nas_server_id:
                self.module.fail_json(msg='Please provide nas server id/name to create CIFS server.')
            if any([netbios_name, workgroup, local_password]) and (not all([netbios_name, workgroup, local_password])):
                msg = 'netbios_name, workgroup and local_password are required to create standalone CIFS server.'
                LOG.error(msg)
                self.module.fail_json(msg=msg)
            result['changed'] = self.create_cifs_server(nas_server_id, interfaces, netbios_name, cifs_server_name, domain, domain_username, domain_password, workgroup, local_password)
        if state == 'absent' and cifs_server_details:
            skip_unjoin = None
            if unjoin_cifs_server_account is not None:
                skip_unjoin = not unjoin_cifs_server_account
            result['changed'] = self.delete_cifs_server(cifs_server_id, skip_unjoin, domain_username, domain_password)
        if state == 'present':
            result['cifs_server_details'] = self.get_details(cifs_server_id=cifs_server_id, cifs_server_name=cifs_server_name, netbios_name=netbios_name, nas_server_id=nas_server_id)
            LOG.info('Process Dict: %s', result['cifs_server_details'])
        self.module.exit_json(**result)
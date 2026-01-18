from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
class QuotaTree(object):
    """Class with Quota Tree operations"""

    def __init__(self):
        """Define all parameters required by this module"""
        self.module_params = utils.get_unity_management_host_parameters()
        self.module_params.update(get_quota_tree_parameters())
        mutually_exclusive = [['filesystem_name', 'filesystem_id'], ['nas_server_name', 'nas_server_id']]
        self.module = AnsibleModule(argument_spec=self.module_params, supports_check_mode=False, mutually_exclusive=mutually_exclusive)
        utils.ensure_required_libs(self.module)
        self.unity_conn = utils.get_unity_unisphere_connection(self.module.params, application_type)

    def check_quota_tree_is_present(self, fs_id, path, tree_quota_id):
        """
            Check if quota tree is present in filesystem.
            :param fs_id: ID of filesystem where quota tree is searched.
            :param path: Path to the quota tree
            :param tree_quota_id: ID of the quota tree
            :return: ID of quota tree if it exists else None.
        """
        if tree_quota_id is None and path is None:
            return None
        all_tree_quota = self.unity_conn.get_tree_quota(filesystem=fs_id, id=tree_quota_id, path=path)
        if tree_quota_id and len(all_tree_quota) == 0 and (self.module.params['state'] == 'present'):
            errormsg = 'Tree quota %s does not exist.' % tree_quota_id
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        if len(all_tree_quota) > 0:
            msg = 'Quota tree with id %s is present in filesystem %s' % (all_tree_quota[0].id, fs_id)
            LOG.info(msg)
            return all_tree_quota[0].id
        else:
            return None

    def create_quota_tree(self, fs_id, soft_limit, hard_limit, unit, path, description):
        """
            Create quota tree of a filesystem.
            :param fs_id: ID of filesystem where quota tree is to be created.
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :param path: Path to quota tree
            :param description: Description for quota tree
            :return: Dict containing new quota tree details.
        """
        if soft_limit is None and hard_limit is None:
            errormsg = 'Both soft limit and hard limit cannot be empty. Please provide atleast one to create quota tree.'
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
        soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
        hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
        try:
            obj_tree_quota = self.unity_conn.create_tree_quota(filesystem_id=fs_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes, path=path, description=description)
            LOG.info('Successfully created quota tree')
            if obj_tree_quota:
                return obj_tree_quota
            else:
                return None
        except Exception as e:
            errormsg = 'Create quota tree operation at path {0} failed in filesystem {1} with error {2}'.format(path, fs_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_filesystem_tree_quota_display_attributes(self, tree_quota_id):
        """Display quota tree attributes
            :param tree_quota_id: Quota tree ID
            :return: Quota tree dict to display
        """
        try:
            tree_quota_obj = self.unity_conn.get_tree_quota(_id=tree_quota_id)
            tree_quota_details = tree_quota_obj._get_properties()
            if tree_quota_obj and tree_quota_obj.existed:
                tree_quota_details['soft_limit'] = utils.convert_size_with_unit(int(tree_quota_details['soft_limit']))
                tree_quota_details['hard_limit'] = utils.convert_size_with_unit(int(tree_quota_details['hard_limit']))
                tree_quota_details['filesystem']['UnityFileSystem']['name'] = tree_quota_obj.filesystem.name
                tree_quota_details['filesystem']['UnityFileSystem'].update({'nas_server': {'name': tree_quota_obj.filesystem.nas_server.name, 'id': tree_quota_obj.filesystem.nas_server.id}})
                return tree_quota_details
        except Exception as e:
            errormsg = 'Failed to display quota tree details {0} with error {1}'.format(tree_quota_obj.id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def get_filesystem(self, nas_server=None, name=None, id=None):
        """
            Get filesystem details.
            :param nas_server: Nas server object.
            :param name: Name of filesystem.
            :param id: ID of filesystem.
            :return: Dict containing filesystem details if it exists.
        """
        id_or_name = id if id else name
        try:
            obj_fs = None
            if name:
                if not nas_server:
                    err_msg = 'NAS Server is required to get the FileSystem.'
                    LOG.error(err_msg)
                    self.module.fail_json(msg=err_msg)
                obj_fs = self.unity_conn.get_filesystem(name=name, nas_server=nas_server)
                if obj_fs and obj_fs.existed:
                    LOG.info('Successfully got the filesystem object %s.', obj_fs)
                    return obj_fs
            if id:
                if nas_server:
                    obj_fs = self.unity_conn.get_filesystem(id=id, nas_server=nas_server)
                else:
                    obj_fs = self.unity_conn.get_filesystem(id=id)
                if obj_fs and obj_fs.existed:
                    LOG.info('Successfully got the filesystem object %s.', obj_fs)
                    return obj_fs
        except Exception as e:
            error_msg = 'Failed to get filesystem %s with error %s.' % (id_or_name, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def get_nas_server_obj(self, name=None, id=None):
        """
            Get nas server details.
            :param name: Nas server name.
            :param id: Nas server ID.
            :return: Dict containing nas server details if it exists.
        """
        nas_server = id if id else name
        error_msg = 'Failed to get NAS server %s.' % nas_server
        try:
            obj_nas = self.unity_conn.get_nas_server(_id=id, name=name)
            if name and obj_nas.existed:
                LOG.info('Successfully got the NAS server object %s.', obj_nas)
                return obj_nas
            elif id and obj_nas.existed:
                LOG.info('Successfully got the NAS server object %s.', obj_nas)
                return obj_nas
            else:
                LOG.error(error_msg)
                self.module.fail_json(msg=error_msg)
        except Exception as e:
            error_msg = 'Failed to get NAS server %s with error %s.' % (nas_server, str(e))
            LOG.error(error_msg)
            self.module.fail_json(msg=error_msg)

    def modify_tree_quota(self, tree_quota_id, soft_limit, hard_limit, unit, description):
        """
            Modify quota tree of filesystem.
            :param tree_quota_id: ID of the quota tree
            :param soft_limit: Soft limit
            :param hard_limit: Hard limit
            :param unit: Unit of soft limit and hard limit
            :param description: Description of quota tree
            :return: Boolean value whether modify quota tree operation is successful.
        """
        try:
            if soft_limit is None and hard_limit is None:
                return False
            tree_quota_obj = self.unity_conn.get_tree_quota(tree_quota_id)._get_properties()
            if soft_limit is None:
                soft_limit_in_bytes = tree_quota_obj['soft_limit']
            else:
                soft_limit_in_bytes = utils.get_size_bytes(soft_limit, unit)
            if hard_limit is None:
                hard_limit_in_bytes = tree_quota_obj['hard_limit']
            else:
                hard_limit_in_bytes = utils.get_size_bytes(hard_limit, unit)
            if description is None:
                description = tree_quota_obj['description']
            if tree_quota_obj:
                if tree_quota_obj['soft_limit'] == soft_limit_in_bytes and tree_quota_obj['hard_limit'] == hard_limit_in_bytes and (tree_quota_obj['description'] == description):
                    return False
                else:
                    modify_tree_quota = self.unity_conn.modify_tree_quota(tree_quota_id=tree_quota_id, hard_limit=hard_limit_in_bytes, soft_limit=soft_limit_in_bytes, description=description)
                    LOG.info('Successfully modified quota tree')
                    if modify_tree_quota:
                        return True
        except Exception as e:
            errormsg = 'Modify quota tree operation {0} failed with error {1}'.format(tree_quota_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def delete_tree_quota(self, tree_quota_id):
        """
        Delete quota tree of a filesystem.
        :param tree_quota_id: ID of quota tree
        :return: Boolean whether quota tree is deleted
        """
        try:
            delete_tree_quota_obj = self.unity_conn.delete_tree_quota(tree_quota_id=tree_quota_id)
            if delete_tree_quota_obj:
                return True
        except Exception as e:
            errormsg = 'Delete operation of quota tree id:{0} failed with error {1}'.format(tree_quota_id, str(e))
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)

    def perform_module_operation(self):
        """
        Perform different actions on quota tree module based on parameters
        passed in the playbook
        """
        filesystem_id = self.module.params['filesystem_id']
        filesystem_name = self.module.params['filesystem_name']
        nas_server_name = self.module.params['nas_server_name']
        nas_server_id = self.module.params['nas_server_id']
        cap_unit = self.module.params['cap_unit']
        state = self.module.params['state']
        hard_limit = self.module.params['hard_limit']
        soft_limit = self.module.params['soft_limit']
        path = self.module.params['path']
        description = self.module.params['description']
        tree_quota_id = self.module.params['tree_quota_id']
        create_tree_quota_obj = None
        nas_server_resource = None
        fs_id = None
        '\n        result is a dictionary to contain end state and quota tree details\n        '
        result = dict(changed=False, create_tree_quota=False, modify_tree_quota=False, get_tree_quota_details={}, delete_tree_quota=False)
        if (soft_limit or hard_limit) and cap_unit is None:
            cap_unit = 'GB'
        if soft_limit and utils.is_size_negative(soft_limit):
            error_message = 'Invalid soft_limit provided, must be greater than or equal to 0'
            LOG.error(error_message)
            self.module.fail_json(msg=error_message)
        if hard_limit and utils.is_size_negative(hard_limit):
            error_message = 'Invalid hard_limit provided, must be greater than or equal to 0'
            LOG.error(error_message)
            self.module.fail_json(msg=error_message)
        '\n        Get NAS server Object\n        '
        if nas_server_name is not None:
            if utils.is_input_empty(nas_server_name):
                self.module.fail_json(msg='Invalid nas_server_name given, Please provide a valid name.')
            nas_server_resource = self.get_nas_server_obj(name=nas_server_name)
        elif nas_server_id is not None:
            if utils.is_input_empty(nas_server_id):
                self.module.fail_json(msg='Invalid nas_server_id given, Please provide a valid ID.')
            nas_server_resource = self.get_nas_server_obj(id=nas_server_id)
        '\n            Get filesystem Object\n        '
        if filesystem_name is not None:
            if utils.is_input_empty(filesystem_name):
                self.module.fail_json(msg='Invalid filesystem_name given, Please provide a valid name.')
            filesystem_obj = self.get_filesystem(nas_server=nas_server_resource, name=filesystem_name)
            fs_id = filesystem_obj.id
        elif filesystem_id is not None:
            if utils.is_input_empty(filesystem_id):
                self.module.fail_json(msg='Invalid filesystem_id given, Please provide a valid ID.')
            filesystem_obj = self.get_filesystem(id=filesystem_id)
            if filesystem_obj:
                fs_id = filesystem_obj[0].id
            else:
                self.module.fail_json(msg='Filesystem does not exist.')
        '\n        Validate path to quota tree\n        '
        if path is not None:
            if utils.is_input_empty(path):
                self.module.fail_json(msg=' Please provide a valid path.')
            elif not path.startswith('/'):
                self.module.fail_json(msg="The path is relative to the root of the file system and must start with a forward slash '/'.")
            if filesystem_id is None and filesystem_name is None:
                self.module.fail_json(msg='Please provide either filesystem_name or fileystem_id.')
        quota_tree_id_present = self.check_quota_tree_is_present(fs_id, path, tree_quota_id)
        tree_quota_id = quota_tree_id_present
        '\n        Create quota tree\n        '
        if (filesystem_id or filesystem_name) and path is not None and (state == 'present'):
            if not tree_quota_id:
                LOG.info('Creating quota tree')
                create_tree_quota_obj = self.create_quota_tree(fs_id, soft_limit, hard_limit, cap_unit, path, description)
        if create_tree_quota_obj:
            tree_quota_id = create_tree_quota_obj.id
            result['create_tree_quota'] = True
        '\n        Modify quota tree\n        '
        if tree_quota_id and state == 'present':
            LOG.info('Modifying quota tree')
            result['modify_tree_quota'] = self.modify_tree_quota(tree_quota_id, soft_limit, hard_limit, cap_unit, description)
        '\n        Delete quota tree\n        '
        if tree_quota_id is not None and state == 'absent':
            LOG.info('Deleting quota tree')
            result['delete_tree_quota'] = self.delete_tree_quota(tree_quota_id)
        '\n        Get quota tree details\n        '
        if state == 'present' and tree_quota_id is not None:
            result['get_tree_quota_details'] = self.get_filesystem_tree_quota_display_attributes(tree_quota_id)
        else:
            result['get_tree_quota_details'] = {}
        if result['create_tree_quota'] or result['modify_tree_quota'] or result['delete_tree_quota']:
            result['changed'] = True
        self.module.exit_json(**result)
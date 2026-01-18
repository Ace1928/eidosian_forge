from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, wait_for_task, get_all_objs
from ansible.module_utils._text import to_native
class VmwareFolderManager(PyVmomi):

    def __init__(self, module):
        super(VmwareFolderManager, self).__init__(module)
        datacenter_name = self.params.get('datacenter', None)
        self.datacenter_obj = find_datacenter_by_name(self.content, datacenter_name=datacenter_name)
        if self.datacenter_obj is None:
            self.module.fail_json(msg='Failed to find datacenter %s' % datacenter_name)
        self.datacenter_folder_type = {'vm': self.datacenter_obj.vmFolder, 'host': self.datacenter_obj.hostFolder, 'datastore': self.datacenter_obj.datastoreFolder, 'network': self.datacenter_obj.networkFolder}

    def ensure(self):
        """
        Manage internal state management
        """
        state = self.module.params.get('state')
        folder_type = self.module.params.get('folder_type')
        folder_name = self.module.params.get('folder_name')
        parent_folder = self.module.params.get('parent_folder', None)
        results = {'changed': False, 'result': {}}
        if state == 'present':
            p_folder_obj = None
            if parent_folder:
                if '/' in parent_folder:
                    parent_folder_parts = parent_folder.strip('/').split('/')
                    p_folder_obj = None
                    for part in parent_folder_parts:
                        part_folder_obj = self.get_folder(folder_name=part, folder_type=folder_type, parent_folder=p_folder_obj)
                        if not part_folder_obj:
                            self.module.fail_json(msg='Could not find folder %s' % part)
                        p_folder_obj = part_folder_obj
                    child_folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, parent_folder=p_folder_obj)
                    if child_folder_obj:
                        results['result'] = 'Folder %s already exists under parent folder %s' % (folder_name, parent_folder)
                        self.module.exit_json(**results)
                else:
                    p_folder_obj = self.get_folder(folder_name=parent_folder, folder_type=folder_type)
                    if not p_folder_obj:
                        self.module.fail_json(msg='Parent folder %s does not exist' % parent_folder)
                    child_folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, parent_folder=p_folder_obj)
                    if child_folder_obj:
                        results['result']['path'] = self.get_folder_path(child_folder_obj)
                        results['result'] = 'Folder %s already exists under parent folder %s' % (folder_name, parent_folder)
                        self.module.exit_json(**results)
            else:
                folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, recurse=True)
                if folder_obj:
                    results['result']['path'] = self.get_folder_path(folder_obj)
                    results['result']['msg'] = 'Folder %s already exists' % folder_name
                    self.module.exit_json(**results)
            try:
                if parent_folder and p_folder_obj:
                    if self.module.check_mode:
                        results['msg'] = "Folder '%s' of type '%s' under '%s' will be created." % (folder_name, folder_type, parent_folder)
                    else:
                        new_folder = p_folder_obj.CreateFolder(folder_name)
                        results['result']['path'] = self.get_folder_path(new_folder)
                        results['result']['msg'] = "Folder '%s' of type '%s' under '%s' created successfully." % (folder_name, folder_type, parent_folder)
                    results['changed'] = True
                elif not parent_folder and (not p_folder_obj):
                    if self.module.check_mode:
                        results['msg'] = "Folder '%s' of type '%s' will be created." % (folder_name, folder_type)
                    else:
                        new_folder = self.datacenter_folder_type[folder_type].CreateFolder(folder_name)
                        results['result']['msg'] = "Folder '%s' of type '%s' created successfully." % (folder_name, folder_type)
                        results['result']['path'] = self.get_folder_path(new_folder)
                    results['changed'] = True
            except vim.fault.DuplicateName as duplicate_name:
                results['changed'] = False
                results['msg'] = 'Failed to create folder as another object has same name in the same target folder : %s' % to_native(duplicate_name.msg)
            except vim.fault.InvalidName as invalid_name:
                self.module.fail_json(msg='Failed to create folder as folder name is not a valid entity name : %s' % to_native(invalid_name.msg))
            except Exception as general_exc:
                self.module.fail_json(msg='Failed to create folder due to generic exception : %s ' % to_native(general_exc))
            self.module.exit_json(**results)
        elif state == 'absent':
            p_folder_obj = None
            if parent_folder:
                if '/' in parent_folder:
                    parent_folder_parts = parent_folder.strip('/').split('/')
                    p_folder_obj = None
                    for part in parent_folder_parts:
                        part_folder_obj = self.get_folder(folder_name=part, folder_type=folder_type, parent_folder=p_folder_obj)
                        if not part_folder_obj:
                            self.module.fail_json(msg='Could not find folder %s' % part)
                        p_folder_obj = part_folder_obj
                    folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, parent_folder=p_folder_obj)
                else:
                    p_folder_obj = self.get_folder(folder_name=parent_folder, folder_type=folder_type)
                    if not p_folder_obj:
                        self.module.fail_json(msg='Parent folder %s does not exist' % parent_folder)
                    folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, parent_folder=p_folder_obj)
            else:
                folder_obj = self.get_folder(folder_name=folder_name, folder_type=folder_type, recurse=True)
            if folder_obj:
                try:
                    if parent_folder:
                        if self.module.check_mode:
                            results['changed'] = True
                            results['msg'] = "Folder '%s' of type '%s' under '%s' will be removed." % (folder_name, folder_type, parent_folder)
                        else:
                            if folder_type == 'vm':
                                task = folder_obj.UnregisterAndDestroy()
                            else:
                                task = folder_obj.Destroy()
                            results['changed'], results['msg'] = wait_for_task(task=task)
                    elif self.module.check_mode:
                        results['changed'] = True
                        results['msg'] = "Folder '%s' of type '%s' will be removed." % (folder_name, folder_type)
                    else:
                        if folder_type == 'vm':
                            task = folder_obj.UnregisterAndDestroy()
                        else:
                            task = folder_obj.Destroy()
                        results['changed'], results['msg'] = wait_for_task(task=task)
                except vim.fault.ConcurrentAccess as concurrent_access:
                    self.module.fail_json(msg='Failed to remove folder as another client modified folder before this operation : %s' % to_native(concurrent_access.msg))
                except vim.fault.InvalidState as invalid_state:
                    self.module.fail_json(msg='Failed to remove folder as folder is in invalid state : %s' % to_native(invalid_state.msg))
                except Exception as gen_exec:
                    self.module.fail_json(msg='Failed to remove folder due to generic exception %s ' % to_native(gen_exec))
            self.module.exit_json(**results)

    def get_folder(self, folder_name, folder_type, parent_folder=None, recurse=False):
        """
        Get managed object of folder by name
        Returns: Managed object of folder by name

        """
        parent_folder = parent_folder or self.datacenter_folder_type[folder_type]
        folder_objs = get_all_objs(self.content, [vim.Folder], parent_folder, recurse=recurse)
        for folder in folder_objs:
            if folder.name == folder_name:
                return folder
        return None
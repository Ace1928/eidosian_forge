from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils import urls
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareGuestFileManager(PyVmomi):

    def __init__(self, module):
        super(VmwareGuestFileManager, self).__init__(module)
        datacenter_name = module.params['datacenter']
        cluster_name = module.params['cluster']
        folder = module.params['folder']
        self.timeout = module.params['timeout']
        datacenter = None
        if datacenter_name:
            datacenter = find_datacenter_by_name(self.content, datacenter_name)
            if not datacenter:
                module.fail_json(msg='Unable to find %(datacenter)s datacenter' % module.params)
        cluster = None
        if cluster_name:
            cluster = find_cluster_by_name(self.content, cluster_name, datacenter)
            if not cluster:
                module.fail_json(msg='Unable to find %(cluster)s cluster' % module.params)
        if module.params['vm_id_type'] == 'inventory_path':
            vm = find_vm_by_id(self.content, vm_id=module.params['vm_id'], vm_id_type='inventory_path', folder=folder)
        else:
            vm = find_vm_by_id(self.content, vm_id=module.params['vm_id'], vm_id_type=module.params['vm_id_type'], datacenter=datacenter, cluster=cluster)
        if not vm:
            module.fail_json(msg='Unable to find virtual machine.')
        self.vm = vm
        try:
            result = dict(changed=False)
            if module.params['directory']:
                result = self.directory()
            if module.params['copy']:
                result = self.copy()
            if module.params['fetch']:
                result = self.fetch()
            module.exit_json(**result)
        except vmodl.RuntimeFault as runtime_fault:
            module.fail_json(msg=to_native(runtime_fault.msg))
        except vmodl.MethodFault as method_fault:
            module.fail_json(msg=to_native(method_fault.msg))
        except Exception as e:
            module.fail_json(msg=to_native(e))

    def directory(self):
        result = dict(changed=True, uuid=self.vm.summary.config.uuid)
        vm_username = self.module.params['vm_username']
        vm_password = self.module.params['vm_password']
        recurse = bool(self.module.params['directory']['recurse'])
        operation = self.module.params['directory']['operation']
        path = self.module.params['directory']['path']
        prefix = self.module.params['directory']['prefix']
        suffix = self.module.params['directory']['suffix']
        creds = vim.vm.guest.NamePasswordAuthentication(username=vm_username, password=vm_password)
        file_manager = self.content.guestOperationsManager.fileManager
        if operation in ('create', 'mktemp'):
            try:
                if operation == 'create':
                    file_manager.MakeDirectoryInGuest(vm=self.vm, auth=creds, directoryPath=path, createParentDirectories=recurse)
                else:
                    newdir = file_manager.CreateTemporaryDirectoryInGuest(vm=self.vm, auth=creds, prefix=prefix, suffix=suffix)
                    result['dir'] = newdir
            except vim.fault.FileAlreadyExists as file_already_exists:
                result['changed'] = False
                result['msg'] = 'Guest directory %s already exist: %s' % (path, to_native(file_already_exists.msg))
            except vim.fault.GuestPermissionDenied as permission_denied:
                self.module.fail_json(msg='Permission denied for path %s : %s' % (path, to_native(permission_denied.msg)), uuid=self.vm.summary.config.uuid)
            except vim.fault.InvalidGuestLogin as invalid_guest_login:
                self.module.fail_json(msg='Invalid guest login for user %s : %s' % (vm_username, to_native(invalid_guest_login.msg)), uuid=self.vm.summary.config.uuid)
            except Exception as e:
                self.module.fail_json(msg='Failed to Create directory into VM VMware exception : %s' % to_native(e), uuid=self.vm.summary.config.uuid)
        if operation == 'delete':
            try:
                file_manager.DeleteDirectoryInGuest(vm=self.vm, auth=creds, directoryPath=path, recursive=recurse)
            except vim.fault.FileNotFound as file_not_found:
                result['changed'] = False
                result['msg'] = 'Guest directory %s not exists %s' % (path, to_native(file_not_found.msg))
            except vim.fault.FileFault as e:
                self.module.fail_json(msg='FileFault : %s' % e.msg, uuid=self.vm.summary.config.uuid)
            except vim.fault.GuestPermissionDenied as permission_denied:
                self.module.fail_json(msg='Permission denied for path %s : %s' % (path, to_native(permission_denied.msg)), uuid=self.vm.summary.config.uuid)
            except vim.fault.InvalidGuestLogin as invalid_guest_login:
                self.module.fail_json(msg='Invalid guest login for user %s : %s' % (vm_username, to_native(invalid_guest_login.msg)), uuid=self.vm.summary.config.uuid)
            except Exception as e:
                self.module.fail_json(msg='Failed to Delete directory into Vm VMware exception : %s' % to_native(e), uuid=self.vm.summary.config.uuid)
        return result

    def fetch(self):
        result = dict(changed=True, uuid=self.vm.summary.config.uuid)
        vm_username = self.module.params['vm_username']
        vm_password = self.module.params['vm_password']
        hostname = self.module.params['hostname']
        dest = self.module.params['fetch']['dest']
        src = self.module.params['fetch']['src']
        creds = vim.vm.guest.NamePasswordAuthentication(username=vm_username, password=vm_password)
        file_manager = self.content.guestOperationsManager.fileManager
        try:
            fileTransferInfo = file_manager.InitiateFileTransferFromGuest(vm=self.vm, auth=creds, guestFilePath=src)
            url = fileTransferInfo.url
            url = url.replace('*', hostname)
            resp, info = urls.fetch_url(self.module, url, method='GET', timeout=self.timeout)
            if info.get('status') != 200 or not resp:
                self.module.fail_json(msg='Failed to fetch file : %s' % info.get('msg', ''), body=info.get('body', ''))
            try:
                with open(dest, 'wb') as local_file:
                    local_file.write(resp.read())
            except Exception as e:
                self.module.fail_json(msg='local file write exception : %s' % to_native(e), uuid=self.vm.summary.config.uuid)
        except vim.fault.FileNotFound as file_not_found:
            self.module.fail_json(msg='Guest file %s does not exist : %s' % (src, to_native(file_not_found.msg)), uuid=self.vm.summary.config.uuid)
        except vim.fault.FileFault as e:
            self.module.fail_json(msg='FileFault : %s' % to_native(e.msg), uuid=self.vm.summary.config.uuid)
        except vim.fault.GuestPermissionDenied:
            self.module.fail_json(msg='Permission denied to fetch file %s' % src, uuid=self.vm.summary.config.uuid)
        except vim.fault.InvalidGuestLogin:
            self.module.fail_json(msg='Invalid guest login for user %s' % vm_username, uuid=self.vm.summary.config.uuid)
        except Exception as e:
            self.module.fail_json(msg='Failed to Fetch file from Vm VMware exception : %s' % to_native(e), uuid=self.vm.summary.config.uuid)
        return result

    def copy(self):
        result = dict(changed=True, uuid=self.vm.summary.config.uuid)
        vm_username = self.module.params['vm_username']
        vm_password = self.module.params['vm_password']
        hostname = self.module.params['hostname']
        overwrite = self.module.params['copy']['overwrite']
        dest = self.module.params['copy']['dest']
        src = self.module.params['copy']['src']
        b_src = to_bytes(src, errors='surrogate_or_strict')
        if not os.path.exists(b_src):
            self.module.fail_json(msg='Source %s not found' % src)
        if not os.access(b_src, os.R_OK):
            self.module.fail_json(msg='Source %s not readable' % src)
        if os.path.isdir(b_src):
            self.module.fail_json(msg='copy does not support copy of directory: %s' % src)
        data = None
        with open(b_src, 'rb') as local_file:
            data = local_file.read()
        file_size = os.path.getsize(b_src)
        creds = vim.vm.guest.NamePasswordAuthentication(username=vm_username, password=vm_password)
        file_attributes = vim.vm.guest.FileManager.FileAttributes()
        file_manager = self.content.guestOperationsManager.fileManager
        try:
            url = file_manager.InitiateFileTransferToGuest(vm=self.vm, auth=creds, guestFilePath=dest, fileAttributes=file_attributes, overwrite=overwrite, fileSize=file_size)
            url = url.replace('*', hostname)
            resp, info = urls.fetch_url(self.module, url, data=data, method='PUT', timeout=self.timeout)
            status_code = info['status']
            if status_code != 200:
                self.module.fail_json(msg='problem during file transfer, http message:%s' % info, uuid=self.vm.summary.config.uuid)
        except vim.fault.FileAlreadyExists:
            result['changed'] = False
            result['msg'] = 'Guest file %s already exists' % dest
            return result
        except vim.fault.FileFault as e:
            self.module.fail_json(msg='FileFault:%s' % to_native(e.msg), uuid=self.vm.summary.config.uuid)
        except vim.fault.GuestPermissionDenied as permission_denied:
            self.module.fail_json(msg='Permission denied to copy file into destination %s : %s' % (dest, to_native(permission_denied.msg)), uuid=self.vm.summary.config.uuid)
        except vim.fault.InvalidGuestLogin as invalid_guest_login:
            self.module.fail_json(msg='Invalid guest login for user %s : %s' % (vm_username, to_native(invalid_guest_login.msg)))
        except Exception as e:
            self.module.fail_json(msg='Failed to Copy file to Vm VMware exception : %s' % to_native(e), uuid=self.vm.summary.config.uuid)
        return result
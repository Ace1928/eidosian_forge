from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_resource_pool_by_name, \
from ansible.module_utils.basic import AnsibleModule
class VMwareGuestRegisterOperation(PyVmomi):

    def __init__(self, module):
        super(VMwareGuestRegisterOperation, self).__init__(module)
        self.datacenter = module.params['datacenter']
        self.cluster = module.params['cluster']
        self.folder = module.params['folder']
        self.name = module.params['name']
        self.esxi_hostname = module.params['esxi_hostname']
        self.path = module.params['path']
        self.template = module.params['template']
        self.resource_pool = module.params['resource_pool']
        self.state = module.params['state']

    def execute(self):
        result = dict(changed=False)
        datacenter = self.find_datacenter_by_name(self.datacenter)
        if not datacenter:
            self.module.fail_json(msg='Cannot find the specified Datacenter: %s' % self.datacenter)
        dcpath = compile_folder_path_for_object(datacenter)
        if not dcpath.endswith('/'):
            dcpath += '/'
        if self.folder in [None, '', '/']:
            self.module.fail_json(msg="Please specify folder path other than blank or '/'")
        elif self.folder.startswith('/vm'):
            fullpath = '%s%s%s' % (dcpath, self.datacenter, self.folder)
        else:
            fullpath = '%s%s' % (dcpath, self.folder)
        folder_obj = self.content.searchIndex.FindByInventoryPath(inventoryPath='%s' % fullpath)
        if not folder_obj:
            details = {'datacenter': datacenter.name, 'datacenter_path': dcpath, 'folder': self.folder, 'full_search_path': fullpath}
            self.module.fail_json(msg='No folder %s matched in the search path : %s' % (self.folder, fullpath), details=details)
        if self.state == 'present':
            vm_obj = self.get_vm()
            if vm_obj:
                if self.module.check_mode:
                    self.module.exit_json(**result)
                self.module.exit_json(**result)
            elif self.module.check_mode:
                result['changed'] = True
                self.module.exit_json(**result)
            if self.esxi_hostname:
                host_obj = self.find_hostsystem_by_name(self.esxi_hostname)
                if not host_obj:
                    self.module.fail_json(msg='Cannot find the specified ESXi host: %s' % self.esxi_hostname)
            else:
                host_obj = None
            if self.cluster:
                cluster_obj = find_cluster_by_name(self.content, self.cluster, datacenter)
                if not cluster_obj:
                    self.module.fail_json(msg='Cannot find the specified cluster name: %s' % self.cluster)
                resource_pool_obj = cluster_obj.resourcePool
            elif self.resource_pool:
                resource_pool_obj = find_resource_pool_by_name(self.content, self.resource_pool)
                if not resource_pool_obj:
                    self.module.fail_json(msg='Cannot find the specified resource pool: %s' % self.resource_pool)
            else:
                resource_pool_obj = host_obj.parent.resourcePool
            task = folder_obj.RegisterVM_Task(path=self.path, name=self.name, asTemplate=self.template, pool=resource_pool_obj, host=host_obj)
            changed = False
            try:
                changed, info = wait_for_task(task)
            except Exception as task_e:
                self.module.fail_json(msg=to_native(task_e))
            result.update(changed=changed)
            self.module.exit_json(**result)
        if self.state == 'absent':
            vm_obj = self.get_vm()
            if vm_obj:
                if self.module.check_mode:
                    result['changed'] = True
                    self.module.exit_json(**result)
            elif self.module.check_mode:
                self.module.exit_json(**result)
            if vm_obj:
                try:
                    vm_obj.UnregisterVM()
                    result.update(changed=True)
                except Exception as exc:
                    self.module.fail_json(msg=to_native(exc))
            self.module.exit_json(**result)
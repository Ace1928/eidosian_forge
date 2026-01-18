from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, TaskError
from ansible.module_utils._text import to_native
class VMwareDatastoreClusterManager(PyVmomi):

    def __init__(self, module):
        """
        Constructor

        """
        super(VMwareDatastoreClusterManager, self).__init__(module)
        datacenter_name = self.params.get('datacenter_name')
        datacenter_obj = self.find_datacenter_by_name(datacenter_name)
        if not datacenter_obj:
            self.module.fail_json(msg="Failed to find datacenter '%s' required for managing datastore cluster." % datacenter_name)
        self.folder_obj = datacenter_obj.datastoreFolder
        self.datastore_cluster_name = self.params.get('datastore_cluster_name')
        self.datastore_cluster_obj = self.find_datastore_cluster_by_name(self.datastore_cluster_name, datacenter=datacenter_obj)
        if not self.datastore_cluster_obj:
            self.module.fail_json(msg="Failed to find the datastore cluster '%s'" % self.datastore_cluster_name)

    def get_datastore_cluster_children(self):
        """
        Return Datastore from the given datastore cluster object

        """
        return [ds for ds in self.datastore_cluster_obj.childEntity if isinstance(ds, vim.Datastore)]

    def ensure(self):
        """
        Manage internal state of datastore cluster

        """
        changed = False
        results = dict(changed=changed)
        temp_result = dict(previous_datastores=[], current_datastores=[], msg='')
        state = self.module.params.get('state')
        datastores = self.module.params.get('datastores') or []
        datastore_obj_list = []
        dsc_child_obj = self.get_datastore_cluster_children()
        if state == 'present':
            temp_result['previous_datastores'] = [ds.name for ds in dsc_child_obj]
            for datastore_name in datastores:
                datastore_obj = self.find_datastore_by_name(datastore_name)
                if not datastore_obj:
                    self.module.fail_json(msg="Failed to find datastore '%s'" % datastore_name)
                if datastore_obj not in dsc_child_obj and datastore_obj not in datastore_obj_list:
                    datastore_obj_list.append(datastore_obj)
            if self.module.check_mode:
                changed_list = [ds.name for ds in datastore_obj_list]
                temp_result['current_datastores'] = temp_result['previous_datastores'].extend(changed_list)
                temp_result['changed_datastores'] = changed_list
                results['changed'] = len(datastore_obj_list) > 0
                results['datastore_cluster_info'] = temp_result
                self.module.exit_json(**results)
            try:
                if datastore_obj_list:
                    task = self.datastore_cluster_obj.MoveIntoFolder_Task(list=datastore_obj_list)
                    changed, result = wait_for_task(task)
                    temp_result['msg'] = result
                temp_result['changed_datastores'] = [ds.name for ds in datastore_obj_list]
                temp_result['current_datastores'] = [ds.name for ds in self.get_datastore_cluster_children()]
            except TaskError as generic_exc:
                self.module.fail_json(msg=to_native(generic_exc))
            except Exception as task_e:
                self.module.fail_json(msg=to_native(task_e))
        elif state == 'absent':
            temp_result['previous_datastores'] = [ds.name for ds in dsc_child_obj]
            temp_result['current_datastores'] = [ds.name for ds in dsc_child_obj]
            for datastore_name in datastores:
                datastore_obj = self.find_datastore_by_name(datastore_name)
                if not datastore_obj:
                    self.module.fail_json(msg="Failed to find datastore '%s'" % datastore_name)
                if datastore_obj in dsc_child_obj and datastore_obj not in datastore_obj_list:
                    datastore_obj_list.append(datastore_obj)
            if self.module.check_mode:
                changed_list = [ds.name for ds in datastore_obj_list]
                for ds in changed_list:
                    temp_result['current_datastores'].pop(ds)
                temp_result['changed_datastores'] = changed_list
                results['changed'] = len(datastore_obj_list) > 0
                results['datastore_cluster_info'] = temp_result
                self.module.exit_json(**results)
            try:
                if datastore_obj_list:
                    task = self.folder_obj.MoveIntoFolder_Task(list=datastore_obj_list)
                    changed, result = wait_for_task(task)
                    temp_result['msg'] = result
                temp_result['changed_datastores'] = [ds.name for ds in datastore_obj_list]
                temp_result['current_datastores'] = [ds.name for ds in self.get_datastore_cluster_children()]
            except TaskError as generic_exc:
                self.module.fail_json(msg=to_native(generic_exc))
            except Exception as task_e:
                self.module.fail_json(msg=to_native(task_e))
        results['changed'] = changed
        results['datastore_cluster_info'] = temp_result
        self.module.exit_json(**results)
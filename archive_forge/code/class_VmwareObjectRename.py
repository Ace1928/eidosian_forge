from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, wait_for_task
class VmwareObjectRename(VmwareRestClient):

    def __init__(self, module):
        """
        Constructor
        """
        super(VmwareObjectRename, self).__init__(module)
        self.pyv = PyVmomi(module=module)
        self.soap_stub = self.pyv.si._stub
        self.object_type = self.params.get('object_type')
        self.object_name = self.params.get('object_name')
        self.object_new_name = self.params.get('new_name')
        self.object_moid = self.params.get('object_moid')
        self.managed_object = None

    def ensure_state(self):
        """
        Manage the internal state of object rename operation

        """
        results = dict(changed=False, rename_status=dict())
        results['rename_status']['desired_name'] = self.object_new_name
        changed = False
        vcenter_obj = self.api_client.vcenter
        available_object_types = [i for i in dir(vcenter_obj) if hasattr(getattr(vcenter_obj, i), 'list') and i != 'Host']
        available_object_types += ['ClusterComputeResource', 'VirtualMachine']
        if self.object_type not in available_object_types:
            self.module.fail_json(msg='Object type can be any one of [%s]' % ', '.join(available_object_types))
        valid_object_types = {'ClusterComputeResource': [vcenter_obj.Cluster, vim.ClusterComputeResource, 'cluster'], 'Cluster': [vcenter_obj.Cluster, vim.ClusterComputeResource, 'cluster'], 'Datacenter': [vcenter_obj.Datacenter, vim.Datacenter, 'datacenter'], 'Datastore': [vcenter_obj.Datastore, vim.Datastore, 'datastore'], 'Folder': [vcenter_obj.Folder, vim.Folder, 'folder'], 'Network': [vcenter_obj.Network, vim.ClusterComputeResource, 'network'], 'ResourcePool': [vcenter_obj.ResourcePool, vim.ResourcePool, 'resource_pool'], 'VM': [vcenter_obj.VM, vim.VirtualMachine, 'vm'], 'VirtualMachine': [vcenter_obj.VM, vim.VirtualMachine, 'vm']}
        target_object = valid_object_types[self.object_type][0]
        filter_spec = target_object.FilterSpec()
        if self.object_moid:
            if target_object is vcenter_obj.Datacenter:
                filter_spec.datacenters = set([self.object_moid])
            if target_object is vcenter_obj.Cluster:
                filter_spec.clusters = set([self.object_moid])
            if target_object is vcenter_obj.ResourcePool:
                filter_spec.resource_pools = set([self.object_moid])
            if target_object is vcenter_obj.Folder:
                filter_spec.folders = set([self.object_moid])
            if target_object is vcenter_obj.VM:
                filter_spec.vms = set([self.object_moid])
            if target_object is vcenter_obj.Network:
                filter_spec.networks = set([self.object_moid])
            if target_object is vcenter_obj.Datastore:
                filter_spec.datastores = set([self.object_moid])
        else:
            filter_spec.names = set([self.object_name])
        all_vmware_objs = target_object.list(filter_spec)
        existing_obj_moid = None
        if self.object_moid:
            if all_vmware_objs:
                if all_vmware_objs[0].name == self.object_new_name:
                    existing_obj_moid = all_vmware_objs
        else:
            existing_obj_moid = target_object.list(target_object.FilterSpec(names=set([self.object_new_name])))
        if existing_obj_moid:
            results['rename_status']['current_name'] = results['rename_status']['previous_name'] = self.object_new_name
            results['changed'] = False
            self.module.exit_json(**results)
        if not all_vmware_objs:
            msg = "Failed to find object with %s '%s' and '%s' object type"
            if self.object_name:
                msg = msg % ('name', self.object_name, self.object_type)
            elif self.object_moid:
                msg = msg % ('moid', self.object_moid, self.object_type)
            self.module.fail_json(msg=msg)
        obj_moid = getattr(all_vmware_objs[0], valid_object_types[self.object_type][2])
        vmware_obj = valid_object_types[self.object_type][1](obj_moid, self.soap_stub)
        if not vmware_obj:
            msg = 'Failed to create VMware object with object %s %s'
            if self.object_name:
                msg = msg % ('name', self.object_name)
            elif self.object_moid:
                msg = msg % ('moid', self.object_moid)
            self.module.fail_json(msg=msg)
        try:
            results['rename_status']['previous_name'] = vmware_obj.name
            if not self.module.check_mode:
                task = vmware_obj.Rename_Task(self.object_new_name)
                wait_for_task(task)
            changed = True
            results['rename_status']['current_name'] = vmware_obj.name
        except Exception as e:
            msg = to_native(e)
            if hasattr(e, 'msg'):
                msg = to_native(e.msg)
            self.module.fail_json(msg=msg)
        results['changed'] = changed
        self.module.exit_json(**results)
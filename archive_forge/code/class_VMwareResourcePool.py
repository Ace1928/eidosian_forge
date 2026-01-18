from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
class VMwareResourcePool(PyVmomi):

    def __init__(self, module):
        super(VMwareResourcePool, self).__init__(module)
        self.datacenter = module.params['datacenter']
        self.resource_pool = module.params['resource_pool']
        self.hostname = module.params['hostname']
        self.username = module.params['username']
        self.password = module.params['password']
        self.state = module.params['state']
        self.mem_shares = module.params['mem_shares']
        self.mem_allocation_shares = module.params['mem_allocation_shares']
        self.mem_limit = module.params['mem_limit']
        self.mem_reservation = module.params['mem_reservation']
        self.mem_expandable_reservations = module.params['mem_expandable_reservations']
        self.cpu_shares = module.params['cpu_shares']
        self.cpu_allocation_shares = module.params['cpu_allocation_shares']
        self.cpu_limit = module.params['cpu_limit']
        self.cpu_reservation = module.params['cpu_reservation']
        self.cpu_expandable_reservations = module.params['cpu_expandable_reservations']
        self.parent_resource_pool = module.params['parent_resource_pool']
        self.resource_pool_obj = None
        self.dc_obj = find_datacenter_by_name(self.content, self.datacenter)
        if self.dc_obj is None:
            self.module.fail_json(msg='Unable to find datacenter with name %s' % self.datacenter)
        if module.params['cluster']:
            self.compute_resource_obj = find_cluster_by_name(self.content, module.params['cluster'], datacenter=self.dc_obj)
            if self.compute_resource_obj is None:
                self.module.fail_json(msg='Unable to find cluster with name %s' % module.params['cluster'])
        if module.params['esxi_hostname']:
            self.compute_resource_obj = find_object_by_name(self.content, module.params['esxi_hostname'], [vim.ComputeResource], folder=self.dc_obj.hostFolder)
            if self.compute_resource_obj is None:
                self.module.fail_json(msg='Unable to find host with name %s' % module.params['esxi_hostname'])
        if module.params['parent_resource_pool']:
            self.compute_resource_obj = find_resource_pool_by_name(self.content, module.params['parent_resource_pool'])
            if self.compute_resource_obj is None:
                self.module.fail_json(msg='Unable to find resource pool with name %s' % module.params['parent_resource_pool'])

    def select_resource_pool(self):
        pool_obj = None
        resource_pools = get_all_objs(self.content, [vim.ResourcePool], folder=self.compute_resource_obj)
        pool_selections = self.get_obj([vim.ResourcePool], self.resource_pool, return_all=True)
        if pool_selections:
            for p in pool_selections:
                if p in resource_pools:
                    pool_obj = p
                    break
        return pool_obj

    def get_obj(self, vimtype, name, return_all=False):
        obj = list()
        container = self.content.viewManager.CreateContainerView(self.content.rootFolder, vimtype, True)
        for c in container.view:
            if name in [c.name, c._GetMoId()]:
                if return_all is False:
                    return c
                else:
                    obj.append(c)
        if len(obj) > 0:
            return obj
        else:
            return None

    def process_state(self):
        try:
            rp_states = {'absent': {'present': self.state_remove_rp, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_update_existing_pr, 'absent': self.state_add_rp}}
            rp_states[self.state][self.check_rp_state()]()
        except vmodl.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=runtime_fault.msg)
        except vmodl.MethodFault as method_fault:
            self.module.fail_json(msg=method_fault.msg)
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def generate_rp_config(self):
        rp_spec = vim.ResourceConfigSpec()
        cpu_alloc = vim.ResourceAllocationInfo()
        cpu_alloc.expandableReservation = self.cpu_expandable_reservations
        cpu_alloc.limit = self.cpu_limit
        cpu_alloc.reservation = self.cpu_reservation
        cpu_alloc_shares = vim.SharesInfo()
        if self.cpu_shares == 'custom':
            cpu_alloc_shares.shares = self.cpu_allocation_shares
        cpu_alloc_shares.level = self.cpu_shares
        cpu_alloc.shares = cpu_alloc_shares
        rp_spec.cpuAllocation = cpu_alloc
        mem_alloc = vim.ResourceAllocationInfo()
        mem_alloc.limit = self.mem_limit
        mem_alloc.expandableReservation = self.mem_expandable_reservations
        mem_alloc.reservation = self.mem_reservation
        mem_alloc_shares = vim.SharesInfo()
        if self.mem_shares == 'custom':
            mem_alloc_shares.shares = self.mem_allocation_shares
        mem_alloc_shares.level = self.mem_shares
        mem_alloc.shares = mem_alloc_shares
        rp_spec.memoryAllocation = mem_alloc
        return rp_spec

    def generate_rp_config_return_value(self, include_rp_config=False):
        resource_config_return_value = {}
        if include_rp_config:
            resource_config_return_value = self.to_json(self.select_resource_pool().config)
        resource_config_return_value['name'] = self.resource_pool
        return resource_config_return_value

    def state_exit_unchanged(self):
        changed = False
        if self.module.check_mode:
            self.module.exit_json(changed=changed)
        self.module.exit_json(changed=changed, resource_pool_config=self.generate_rp_config_return_value())

    def state_update_existing_pr(self):
        changed = False
        rp_spec = self.generate_rp_config()
        if self.mem_shares and self.mem_shares != self.resource_pool_obj.config.memoryAllocation.shares.level:
            changed = True
            rp_spec.memoryAllocation.shares.level = self.mem_shares
        if self.mem_allocation_shares and self.mem_shares == 'custom':
            if self.mem_allocation_shares != self.resource_pool_obj.config.memoryAllocation.shares.shares:
                changed = True
                rp_spec.memoryAllocation.shares.shares = self.mem_allocation_shares
        if self.mem_limit and self.mem_limit != self.resource_pool_obj.config.memoryAllocation.limit:
            changed = True
            rp_spec.memoryAllocation.limit = self.mem_limit
        if self.mem_reservation and self.mem_reservation != self.resource_pool_obj.config.memoryAllocation.reservation:
            changed = True
            rp_spec.memoryAllocation.reservation = self.mem_reservation
        if self.mem_expandable_reservations != self.resource_pool_obj.config.memoryAllocation.expandableReservation:
            changed = True
            rp_spec.memoryAllocation.expandableReservation = self.mem_expandable_reservations
        if self.cpu_shares and self.cpu_shares != self.resource_pool_obj.config.cpuAllocation.shares.level:
            changed = True
            rp_spec.cpuAllocation.shares.level = self.cpu_shares
        if self.cpu_allocation_shares and self.cpu_shares == 'custom':
            if self.cpu_allocation_shares != self.resource_pool_obj.config.cpuAllocation.shares.shares:
                changed = True
                rp_spec.cpuAllocation.shares.shares = self.cpu_allocation_shares
        if self.cpu_limit and self.cpu_limit != self.resource_pool_obj.config.cpuAllocation.limit:
            changed = True
            rp_spec.cpuAllocation.limit = self.cpu_limit
        if self.cpu_reservation and self.cpu_reservation != self.resource_pool_obj.config.cpuAllocation.reservation:
            changed = True
            rp_spec.cpuAllocation.reservation = self.cpu_reservation
        if self.cpu_expandable_reservations != self.resource_pool_obj.config.cpuAllocation.expandableReservation:
            changed = True
            rp_spec.cpuAllocation.expandableReservation = self.cpu_expandable_reservations
        if self.module.check_mode:
            self.module.exit_json(changed=changed)
        if changed:
            self.resource_pool_obj.UpdateConfig(self.resource_pool, rp_spec)
        resource_pool_config = self.generate_rp_config_return_value(True)
        self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)

    def state_remove_rp(self):
        changed = True
        result = None
        if self.module.check_mode:
            self.module.exit_json(changed=changed)
        resource_pool_config = self.generate_rp_config_return_value(True)
        try:
            task = self.resource_pool_obj.Destroy()
            success, result = wait_for_task(task)
        except Exception:
            self.module.fail_json(msg="Failed to remove resource pool '%s' '%s'" % (self.resource_pool, self.resource_pool))
        self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)

    def state_add_rp(self):
        changed = True
        if self.module.check_mode:
            self.module.exit_json(changed=changed)
        rp_spec = self.generate_rp_config()
        if self.parent_resource_pool:
            rootResourcePool = self.compute_resource_obj
        else:
            rootResourcePool = self.compute_resource_obj.resourcePool
        rootResourcePool.CreateResourcePool(self.resource_pool, rp_spec)
        resource_pool_config = self.generate_rp_config_return_value(True)
        self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)

    def check_rp_state(self):
        self.resource_pool_obj = self.select_resource_pool()
        if self.resource_pool_obj is None:
            return 'absent'
        return 'present'
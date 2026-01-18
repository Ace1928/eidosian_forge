from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class AffinityGroupsModule(BaseModule):

    def __init__(self, vm_ids, host_ids, host_label_ids, vm_label_ids, *args, **kwargs):
        super(AffinityGroupsModule, self).__init__(*args, **kwargs)
        self._vm_ids = vm_ids
        self._host_ids = host_ids
        self._host_label_ids = host_label_ids
        self._vm_label_ids = vm_label_ids

    def update_vms(self, affinity_group):
        """
        This method iterate via the affinity VM assignments and datech the VMs
        which should not be attached to affinity and attach VMs which should be
        attached to affinity.
        """
        assigned_vms = self.assigned_vms(affinity_group)
        to_remove = list((vm for vm in assigned_vms if vm not in self._vm_ids))
        to_add = []
        if self._vm_ids:
            to_add = list((vm for vm in self._vm_ids if vm not in assigned_vms))
        ag_service = self._service.group_service(affinity_group.id)
        for vm in to_remove:
            ag_service.vms_service().vm_service(vm).remove()
        for vm in to_add:
            try:
                ag_service.vms_service().add(otypes.Vm(id=vm))
            except ValueError as ex:
                if 'complete' not in str(ex):
                    raise ex

    def post_create(self, entity):
        self.update_vms(entity)

    def post_update(self, entity):
        self.update_vms(entity)

    def build_entity(self):
        affinity_group = otypes.AffinityGroup(name=self._module.params['name'], description=self._module.params['description'], positive=self._module.params['vm_rule'] == 'positive' if self._module.params['vm_rule'] is not None else None, enforcing=self._module.params['vm_enforcing'] if self._module.params['vm_enforcing'] is not None else None)
        if not engine_supported(self._connection, '4.1'):
            return affinity_group
        affinity_group.hosts_rule = otypes.AffinityRule(positive=self.param('host_rule') == 'positive' if self.param('host_rule') is not None else None, enforcing=self.param('host_enforcing')) if self.param('host_enforcing') is not None or self.param('host_rule') is not None else None
        affinity_group.vms_rule = otypes.AffinityRule(positive=self.param('vm_rule') == 'positive' if self.param('vm_rule') is not None else None, enforcing=self.param('vm_enforcing'), enabled=self.param('vm_rule') in ['negative', 'positive'] if self.param('vm_rule') is not None else None) if self.param('vm_enforcing') is not None or self.param('vm_rule') is not None else None
        affinity_group.hosts = [otypes.Host(id=host_id) for host_id in self._host_ids] if self._host_ids is not None else None
        affinity_group.vm_labels = [otypes.AffinityLabel(id=host_id) for host_id in self._vm_label_ids] if self._vm_label_ids is not None else None
        affinity_group.host_labels = [otypes.AffinityLabel(id=host_id) for host_id in self._host_label_ids] if self._host_label_ids is not None else None
        return affinity_group

    def assigned_vms(self, affinity_group):
        if getattr(affinity_group.vms, 'href', None):
            return sorted([vm.id for vm in self._connection.follow_link(affinity_group.vms)])
        else:
            return sorted([vm.id for vm in affinity_group.vms])

    def update_check(self, entity):
        assigned_vms = self.assigned_vms(entity)
        do_update = equal(self.param('description'), entity.description) and equal(self.param('vm_enforcing'), entity.enforcing) and equal(self.param('vm_rule') == 'positive' if self.param('vm_rule') else None, entity.positive) and equal(self._vm_ids, assigned_vms)
        if not engine_supported(self._connection, '4.1'):
            return do_update
        return do_update and (equal(self.param('host_rule') == 'positive' if self.param('host_rule') else None, entity.hosts_rule.positive) and equal(self.param('host_enforcing'), entity.hosts_rule.enforcing) and equal(self.param('vm_rule') in ['negative', 'positive'] if self.param('vm_rule') else None, entity.vms_rule.enabled) and equal(self._host_ids, sorted([host.id for host in entity.hosts])))
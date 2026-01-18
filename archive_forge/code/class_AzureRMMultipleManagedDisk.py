from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMMultipleManagedDisk(AzureRMModuleBase):
    """Configuration class for an Azure RM Managed Disk resource"""

    def __init__(self):
        managed_by_extended_spec = dict(resource_group=dict(type='str'), name=dict(type='str'))
        managed_disks_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), storage_account_type=dict(type='str', choices=['Standard_LRS', 'StandardSSD_LRS', 'StandardSSD_ZRS', 'Premium_LRS', 'Premium_ZRS', 'UltraSSD_LRS']), create_option=dict(type='str', choices=['empty', 'import', 'copy']), storage_account_id=dict(type='str'), source_uri=dict(type='str', aliases=['source_resource_uri']), os_type=dict(type='str', choices=['linux', 'windows']), disk_size_gb=dict(type='int'), zone=dict(type='str', choices=['', '1', '2', '3']), attach_caching=dict(type='str', choices=['', 'read_only', 'read_write']), lun=dict(type='int'), max_shares=dict(type='int'))
        self.module_arg_spec = dict(state=dict(type='str', default='present', choices=['present', 'absent']), managed_disks=dict(type='list', elements='dict', options=managed_disks_spec), managed_by_extended=dict(type='list', elements='dict', options=managed_by_extended_spec))
        self.results = dict(changed=False, state=list())
        super(AzureRMMultipleManagedDisk, self).__init__(derived_arg_spec=self.module_arg_spec, supports_tags=True)

    def validate_disks_parameter(self):
        errors = []
        create_option_reqs = [('import', ['source_uri', 'storage_account_id']), ('copy', ['source_uri']), ('empty', ['disk_size_gb'])]
        for disk in self.managed_disks:
            create_option = disk.get('create_option')
            for req in create_option_reqs:
                if create_option == req[0] and any((disk.get(opt) is None for opt in req[1])):
                    errors.append('managed disk {0}/{1} has create_option set to {2} but not all required parameters ({3}) are set.'.format(disk.get('resource_group'), disk.get('name'), req[0], ','.join(req[1])))
        if errors:
            self.fail(msg='Some required options are missing from managed disks configuration.', errors=errors)

    def generate_disk_parameters(self, location, tags, zone=None, storage_account_type=None, disk_size_gb=None, create_option=None, source_uri=None, storage_account_id=None, os_type=None, max_shares=None, **kwargs):
        disk_params = {}
        creation_data = {}
        disk_params['location'] = location
        disk_params['tags'] = tags
        if zone:
            disk_params['zones'] = [zone]
        if storage_account_type:
            storage = self.compute_models.DiskSku(name=storage_account_type)
            disk_params['sku'] = storage
        disk_params['disk_size_gb'] = disk_size_gb
        creation_data['create_option'] = self.compute_models.DiskCreateOption.empty
        if create_option == 'import':
            creation_data['create_option'] = self.compute_models.DiskCreateOption.import_enum
            creation_data['source_uri'] = source_uri
            creation_data['storage_account_id'] = storage_account_id
        elif create_option == 'copy':
            creation_data['create_option'] = self.compute_models.DiskCreateOption.copy
            creation_data['source_resource_id'] = source_uri
        if os_type:
            disk_params['os_type'] = self.compute_models.OperatingSystemTypes(os_type.capitalize())
        else:
            disk_params['os_type'] = None
        if max_shares:
            disk_params['max_shares'] = max_shares
        disk_params['creation_data'] = creation_data
        return disk_params

    def get_disk_instance(self, managed_disk):
        resource_group = self.get_resource_group(managed_disk.get('resource_group'))
        managed_disk['location'] = managed_disk.get('location') or resource_group.location
        disk_instance = self.get_managed_disk(resource_group=managed_disk.get('resource_group'), name=managed_disk.get('name'))
        if disk_instance is not None:
            for key in ('create_option', 'source_uri', 'disk_size_gb', 'os_type', 'zone'):
                if managed_disk.get(key) is None:
                    managed_disk[key] = disk_instance.get(key)
        parameter = self.generate_disk_parameters(tags=self.tags, **managed_disk)
        return (parameter, disk_instance)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        self.tags = kwargs.get('tags')
        state = kwargs.get('state')
        self.managed_disks = kwargs.get('managed_disks')
        self.managed_by_extended = kwargs.get('managed_by_extended')
        self.validate_disks_parameter()
        managed_vm_id = []
        if self.managed_by_extended:
            managed_vm_id = [self._get_vm(vm['resource_group'], vm['name']) for vm in self.managed_by_extended]
        if state == 'present':
            return self.create_or_attach_disks(managed_vm_id)
        elif state == 'absent':
            return self.detach_or_delete_disks(managed_vm_id)

    def compute_disks_result(self, disk_instances):
        result = []
        for params, disk in disk_instances:
            disk_id = parse_resource_id(disk.get('id'))
            result.append(self.get_managed_disk(resource_group=disk_id.get('resource_group'), name=disk_id.get('resource_name')))
        return result

    def create_or_attach_disks(self, managed_vm_id):
        changed, disk_instances, disks_to_create = (False, [], [])
        for disk in self.managed_disks:
            parameter, disk_instance = self.get_disk_instance(disk)
            disk_info_to_compare = dict(zone=disk.get('zone'), max_shares=disk.get('max_shares'), found_disk=disk_instance, new_disk=parameter)
            if disk_instance is None or self.is_different(**disk_info_to_compare):
                disks_to_create.append((disk, parameter))
            else:
                disk_instances.append((disk, disk_instance))
        if len(disks_to_create) > 0:
            changed = True
            result = self.create_or_update_disks(disks_to_create)
            disk_instances += result
        if self.managed_by_extended is not None and len(self.managed_by_extended) > 0:
            attach_config = []
            for vm in managed_vm_id:
                time.sleep(5)
                disks = [(d, i) for d, i in disk_instances if not self._is_disk_attached_to_vm(vm.id, i)]
                if len(disks) > 0:
                    attach_config.append(self.create_attachment_configuration(vm, disks))
            if len(attach_config) > 0:
                changed = True
                self.update_virtual_machines(attach_config)
        elif self.managed_by_extended == []:
            changed = self.detach_disks_from_all_vms(disk_instances) or changed
        return dict(changed=changed, state=self.compute_disks_result(disk_instances))

    def detach_or_delete_disks(self, managed_vm_id):
        changed, disk_instances = (False, [])
        for disk in self.managed_disks:
            params, disk_instance = self.get_disk_instance(disk)
            if disk_instance is not None:
                disk_instances.append((disk, disk_instance))
        result = []
        if self.managed_by_extended is not None and len(self.managed_by_extended) > 0:
            disks_names = [d.get('name').lower() for p, d in disk_instances]
            attach_config = []
            for vm in managed_vm_id:
                disks = [d for p, d in disk_instances if self._is_disk_attached_to_vm(vm.id, d)]
                if len(disks) > 0:
                    attach_config.append(self.create_detachment_configuration(vm, disks_names))
            if len(attach_config) > 0:
                changed = True
                self.update_virtual_machines(attach_config)
            result = self.compute_disks_result(disk_instances)
        elif self.managed_by_extended is None:
            changed = self.detach_disks_from_all_vms(disk_instances)
            if len(disk_instances) > 0:
                disks_ids = [disk.get('id') for param, disk in disk_instances]
                changed = True
                self.delete_disks(disks_ids)
        return dict(changed=changed, state=result)

    def detach_disks_from_all_vms(self, disk_instances):
        changed = False
        unique_vm_id = []
        for param, disk_instance in disk_instances:
            managed_by_vm = disk_instance.get('managed_by')
            managed_by_extended_vms = disk_instance.get('managed_by_extended') or []
            if managed_by_vm is not None and managed_by_vm not in unique_vm_id:
                unique_vm_id.append(managed_by_vm)
            for vm_id in managed_by_extended_vms:
                if vm_id not in unique_vm_id:
                    unique_vm_id.append(vm_id)
        if unique_vm_id:
            disks_names = [instance.get('name').lower() for d, instance in disk_instances]
            changed = True
            attach_config = []
            for vm_id in unique_vm_id:
                vm_name_id = parse_resource_id(vm_id)
                vm_instance = self._get_vm(vm_name_id['resource_group'], vm_name_id['resource_name'])
                attach_config.append(self.create_detachment_configuration(vm_instance, disks_names))
            if len(attach_config) > 0:
                changed = True
                self.update_virtual_machines(attach_config)
        return changed

    def _is_disk_attached_to_vm(self, vm_id, item):
        managed_by = item['managed_by']
        managed_by_extended = item['managed_by_extended']
        if managed_by is not None and vm_id == managed_by:
            return True
        if managed_by_extended is not None and vm_id in managed_by_extended:
            return True
        return False

    def create_attachment_configuration(self, vm, disks):
        vm_id = parse_resource_id(vm.id)
        for managed_disk, disk_instance in disks:
            lun = managed_disk.get('lun')
            if lun is None:
                luns = [d.lun for d in vm.storage_profile.data_disks] if vm.storage_profile.data_disks else []
                lun = 0
                while True:
                    if lun not in luns:
                        break
                    lun = lun + 1
                for item in vm.storage_profile.data_disks:
                    if item.name == managed_disk.get('name'):
                        lun = item.lun
            params = self.compute_models.ManagedDiskParameters(id=disk_instance.get('id'), storage_account_type=disk_instance.get('storage_account_type'))
            attach_caching = managed_disk.get('attach_caching')
            caching_options = self.compute_models.CachingTypes[attach_caching] if attach_caching and attach_caching != '' else None
            data_disk = self.compute_models.DataDisk(lun=lun, create_option=self.compute_models.DiskCreateOptionTypes.attach, managed_disk=params, caching=caching_options)
            vm.storage_profile.data_disks.append(data_disk)
        return (vm_id['resource_group'], vm_id['resource_name'], vm)

    def create_detachment_configuration(self, vm_instance, disks_names):
        vm_data = parse_resource_id(vm_instance.id)
        leftovers = [d for d in vm_instance.storage_profile.data_disks if d.name.lower() not in disks_names]
        if len(vm_instance.storage_profile.data_disks) == len(leftovers):
            self.fail("None of the following disks '{0}' are attached to the VM '{1}/{2}'.".format(disks_names, vm_data['resource_group'], vm_data['resource_name']))
        vm_instance.storage_profile.data_disks = leftovers
        return (vm_data['resource_group'], vm_data['resource_name'], vm_instance)

    def _get_vm(self, resource_group, name):
        try:
            return self.compute_client.virtual_machines.get(resource_group, name, expand='instanceview')
        except Exception as exc:
            self.fail('Error getting virtual machine {0}/{1} - {2}'.format(resource_group, name, str(exc)))

    def create_or_update_disks(self, disks_to_create):
        pollers = []
        for disk_info, disk in disks_to_create:
            resource_group = disk_info.get('resource_group')
            name = disk_info.get('name')
            try:
                poller = self.compute_client.disks.begin_create_or_update(resource_group, name, disk)
                pollers.append(poller)
            except Exception as e:
                self.fail('Error creating the managed disk {0}/{1}: {2}'.format(resource_group, name, str(e)))
        disks_instances = self.get_multiple_pollers_results(pollers)
        result = []
        for i, instance in enumerate(disks_instances):
            result.append((disks_to_create[i][0], managed_disk_to_dict(instance)))
        return result

    def is_different(self, zone, max_shares, found_disk, new_disk):
        resp = False
        if new_disk.get('disk_size_gb'):
            if not found_disk['disk_size_gb'] == new_disk['disk_size_gb']:
                resp = True
        if new_disk.get('os_type'):
            if found_disk['os_type'] is None or not self.compute_models.OperatingSystemTypes(found_disk['os_type'].capitalize()) == new_disk['os_type']:
                resp = True
        if new_disk.get('sku'):
            if not found_disk['storage_account_type'] == new_disk['sku'].name:
                resp = True
        if new_disk.get('tags') is not None:
            if not found_disk['tags'] == new_disk['tags']:
                resp = True
        if zone is not None:
            if not found_disk['zone'] == zone:
                resp = True
        if max_shares is not None:
            if not found_disk['max_shares'] == max_shares:
                resp = True
        return resp

    def delete_disks(self, ids):
        pollers = []
        for disk_id in ids:
            try:
                disk = parse_resource_id(disk_id)
                resource_group, name = (disk.get('resource_group'), disk.get('resource_name'))
                poller = self.compute_client.disks.begin_delete(resource_group, name)
                pollers.append(poller)
            except Exception as e:
                self.fail('Error deleting the managed disk {0}/{1}: {2}'.format(resource_group, name, str(e)))
        return self.get_multiple_pollers_results(pollers)

    def update_virtual_machines(self, config):
        pollers = []
        for resource_group, name, params in config:
            try:
                poller = self.compute_client.virtual_machines.begin_create_or_update(resource_group, name, params)
                pollers.append(poller)
            except AzureError as exc:
                self.fail('Error updating virtual machine (attaching/detaching disks) {0}/{1} - {2}'.format(resource_group, name, exc.message))
        return self.get_multiple_pollers_results(pollers)

    def get_managed_disk(self, resource_group, name):
        try:
            resp = self.compute_client.disks.get(resource_group, name)
            return managed_disk_to_dict(resp)
        except ResourceNotFoundError:
            self.log('Did not find managed disk {0}/{1}'.format(resource_group, name))
from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import re
class AzureRMVirtualMachineScaleSetInfo(AzureRMModuleBase):
    """Utility class to get virtual machine scale set facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'), format=dict(type='str', choices=['curated', 'raw'], default='raw'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.format = None
        self.tags = None
        super(AzureRMVirtualMachineScaleSetInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachinescaleset_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachinescaleset_facts' module has been renamed to 'azure_rm_virtualmachinescaleset_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            result = self.get_item()
        else:
            result = self.list_items()
        if self.format == 'curated':
            for index in range(len(result)):
                vmss = result[index]
                subnet_name = None
                load_balancer_name = None
                virtual_network_name = None
                ssh_password_enabled = False
                try:
                    subnet_id = vmss['virtual_machine_profile']['network_profile']['network_interface_configurations'][0]['ipConfigurations'][0]['subnet']['id']
                    subnet_name = re.sub('.*subnets\\/', '', subnet_id)
                except Exception:
                    self.log('Could not extract subnet name')
                try:
                    backend_address_pool_id = vmss['virtual_machine_profile']['network_profile']['network_interface_configurations'][0]['ip_configurations'][0]['load_balancer_backend_address_pools'][0]['id']
                    load_balancer_name = re.sub('\\/backendAddressPools.*', '', re.sub('.*loadBalancers\\/', '', backend_address_pool_id))
                    virtual_network_name = re.sub('.*virtualNetworks\\/', '', re.sub('\\/subnets.*', '', subnet_id))
                except Exception:
                    self.log('Could not extract load balancer / virtual network name')
                try:
                    ssh_password_enabled = not vmss['virtual_machine_profile']['os_profile']['linux_configuration']['disable_password_authentication']
                except Exception:
                    self.log('Could not extract SSH password enabled')
                data_disks = vmss['virtual_machine_profile']['storage_profile'].get('data_disks', [])
                for disk_index in range(len(data_disks)):
                    old_disk = data_disks[disk_index]
                    new_disk = {'lun': old_disk['lun'], 'disk_size_gb': old_disk['disk_size_gb'], 'managed_disk_type': old_disk['managed_disk']['storage_account_type'], 'caching': old_disk['caching']}
                    data_disks[disk_index] = new_disk
                updated = {'id': vmss['id'], 'resource_group': self.resource_group, 'name': vmss['name'], 'state': 'present', 'location': vmss['location'], 'vm_size': vmss['sku']['name'], 'capacity': vmss['sku']['capacity'], 'tier': vmss['sku']['tier'], 'upgrade_policy': vmss.get('upgrade_policy'), 'orchestrationMode': vmss.get('orchestration_mode'), 'platformFaultDomainCount': vmss.get('platform_fault_domain_count'), 'admin_username': vmss['virtual_machine_profile']['os_profile']['admin_username'], 'admin_password': vmss['virtual_machine_profile']['os_profile'].get('admin_password'), 'ssh_password_enabled': ssh_password_enabled, 'image': vmss['virtual_machine_profile']['storage_profile']['image_reference'], 'os_disk_caching': vmss['virtual_machine_profile']['storage_profile']['os_disk']['caching'], 'os_type': 'Linux' if vmss['virtual_machine_profile']['os_profile'].get('linux_configuration') is not None else 'Windows', 'overprovision': vmss.get('overprovision'), 'managed_disk_type': vmss['virtual_machine_profile']['storage_profile']['os_disk']['managed_disk']['storage_account_type'], 'data_disks': data_disks, 'virtual_network_name': virtual_network_name, 'subnet_name': subnet_name, 'load_balancer': load_balancer_name, 'tags': vmss.get('tags')}
                result[index] = updated
        if is_old_facts:
            self.results['ansible_facts'] = {'azure_vmss': result}
            if self.format == 'curated':
                self.results['vmss'] = result
        else:
            self.results['vmss'] = result
        return self.results

    def get_item(self):
        """Get a single virtual machine scale set"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.compute_client.virtual_machine_scale_sets.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [self.serialize_obj(item, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)]
        return results

    def list_items(self):
        """Get all virtual machine scale sets"""
        self.log('List all virtual machine scale sets')
        try:
            response = self.compute_client.virtual_machine_scale_sets.list(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_obj(item, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES))
        return results
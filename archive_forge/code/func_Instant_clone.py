from __future__ import (absolute_import, division, print_function)
import time
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def Instant_clone(self):
    if self.vm_obj is None:
        vm_id = self.parent_vm or self.uuid or self.moid
        self.module.fail_json(msg='Failed to find the VM/template with %s' % vm_id)
    try:
        task = self.vm_obj.InstantClone_Task(spec=self.instant_clone_spec)
        wait_for_task(task)
        vm_info = self.get_new_vm_info(self.vm_name)
        result = {'changed': True, 'failed': False, 'vm_info': vm_info}
    except TaskError as task_e:
        self.module.fail_json(msg=to_native(task_e))
    self.destination_content = connect_to_api(self.module, hostname=self.hostname, username=self.username, password=self.password, port=self.port, validate_certs=self.validate_certs)
    vm_IC = find_vm_by_name(content=self.destination_content, vm_name=self.params['name'])
    if vm_IC and self.params.get('guestinfo_vars'):
        guest_custom_mng = self.destination_content.guestCustomizationManager
        auth_obj = vim.vm.guest.NamePasswordAuthentication()
        guest_user = self.params.get('vm_username')
        guest_password = self.params.get('vm_password')
        auth_obj.username = guest_user
        auth_obj.password = guest_password
        guestinfo_vars = self.params.get('guestinfo_vars')
        customization_spec = vim.vm.customization.Specification()
        customization_spec.globalIPSettings = vim.vm.customization.GlobalIPSettings()
        customization_spec.globalIPSettings.dnsServerList = [guestinfo_vars[0]['dns']]
        customization_spec.identity = vim.vm.customization.LinuxPrep()
        customization_spec.identity.domain = guestinfo_vars[0]['domain']
        customization_spec.identity.hostName = vim.vm.customization.FixedName()
        customization_spec.identity.hostName.name = guestinfo_vars[0]['hostname']
        customization_spec.nicSettingMap = []
        adapter_mapping_obj = vim.vm.customization.AdapterMapping()
        adapter_mapping_obj.adapter = vim.vm.customization.IPSettings()
        adapter_mapping_obj.adapter.ip = vim.vm.customization.FixedIp()
        adapter_mapping_obj.adapter.ip.ipAddress = guestinfo_vars[0]['ipaddress']
        adapter_mapping_obj.adapter.subnetMask = guestinfo_vars[0]['netmask']
        adapter_mapping_obj.adapter.gateway = [guestinfo_vars[0]['gateway']]
        customization_spec.nicSettingMap.append(adapter_mapping_obj)
        try:
            task_guest = guest_custom_mng.CustomizeGuest_Task(vm_IC, auth_obj, customization_spec)
            wait_for_task(task_guest)
            vm_info = self.get_new_vm_info(self.vm_name)
            result = {'changed': True, 'failed': False, 'vm_info': vm_info}
        except TaskError as task_e:
            self.module.fail_json(msg=to_native(task_e))
        instant_vm_obj = find_vm_by_id(content=self.content, vm_id=vm_info['instance_uuid'], vm_id_type='instance_uuid')
        set_vm_power_state(content=self.content, vm=instant_vm_obj, state='rebootguest', force=False)
        if self.wait_vm_tools:
            interval = 15
            while self.wait_vm_tools_timeout > 0:
                if instant_vm_obj.guest.toolsRunningStatus != 'guestToolsRunning':
                    break
                self.wait_vm_tools_timeout -= interval
                time.sleep(interval)
            while self.wait_vm_tools_timeout > 0:
                if instant_vm_obj.guest.toolsRunningStatus == 'guestToolsRunning':
                    break
                self.wait_vm_tools_timeout -= interval
                time.sleep(interval)
            if self.wait_vm_tools_timeout <= 0:
                self.module.fail_json(msg='Timeout has been reached for waiting to start the vm tools.')
    return result
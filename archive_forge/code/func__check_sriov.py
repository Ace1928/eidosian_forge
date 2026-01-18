from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
def _check_sriov(self, host):
    pnic_info = {}
    pnic_info['rebootRequired'] = host.summary.rebootRequired
    for pci_device in host.configManager.pciPassthruSystem.pciPassthruInfo:
        if pci_device.id == self._getPciId(host):
            try:
                if pci_device.sriovCapable:
                    pnic_info['sriovCapable'] = True
                    pnic_info['sriovEnabled'] = pci_device.sriovEnabled
                    pnic_info['sriovActive'] = pci_device.sriovActive
                    pnic_info['numVirtualFunction'] = pci_device.numVirtualFunction
                    pnic_info['numVirtualFunctionRequested'] = pci_device.numVirtualFunctionRequested
                    pnic_info['maxVirtualFunctionSupported'] = pci_device.maxVirtualFunctionSupported
                else:
                    pnic_info['sriovCapable'] = False
            except AttributeError:
                pnic_info['sriovCapable'] = False
            break
    return pnic_info
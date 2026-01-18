from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _get_compute_resource_by_name(self, recurse=True):
    """
        get compute resource object with matching name of esxi_hostname or cluster
        parameters.
        :param recurse: recurse vmware content folder, default is True
        :return: object matching vim.ComputeResource or None if no match
        :rtype: object
        """
    resource_name = None
    if self.params['esxi_hostname']:
        resource_name = self.params['esxi_hostname']
    if self.params['cluster']:
        resource_name = self.params['cluster']
    container = self.content.viewManager.CreateContainerView(self.content.rootFolder, [vim.ComputeResource], recurse)
    for obj in container.view:
        if self.params['esxi_hostname'] and isinstance(obj, vim.ClusterComputeResource) and hasattr(obj, 'host'):
            for host in obj.host:
                if host.name == resource_name:
                    return obj
        if obj.name == resource_name:
            return obj
    return None
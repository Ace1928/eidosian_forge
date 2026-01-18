from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _set_host_obj_list(self, host_list=None):
    """
        Populate host object list from list of hostnames
        Args:
            host_list: List of host names

        Returns: None

        """
    if host_list is None:
        host_list = self._host_list
    if host_list is not None:
        for host in host_list:
            if self.module.check_mode is False:
                host_obj = self.find_hostsystem_by_name(host)
                if host_obj is None and self.module.check_mode is False:
                    self.module.fail_json(msg='ESXi host %s does not exist in cluster %s' % (host, self._cluster_name))
                self._host_obj_list.append(host_obj)
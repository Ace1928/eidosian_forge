from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def _get_vlan(self, network_domain):
    """
        Retrieve the target VLAN details from CloudControl.

        :param network_domain: The target network domain.
        :return: The VLAN, or None if the target VLAN was not found.
        :rtype: DimensionDataVlan
        """
    vlans = self.driver.ex_list_vlans(location=self.location, network_domain=network_domain)
    matching_vlans = [vlan for vlan in vlans if vlan.name == self.name]
    if matching_vlans:
        return matching_vlans[0]
    return None
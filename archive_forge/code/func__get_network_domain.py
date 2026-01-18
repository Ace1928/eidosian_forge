from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def _get_network_domain(self):
    """
        Retrieve the target network domain from the Cloud Control API.

        :return: The network domain.
        """
    try:
        return self.get_network_domain(self.network_domain_selector, self.location)
    except UnknownNetworkError:
        self.module.fail_json(msg='Cannot find network domain "{0}" in datacenter "{1}".'.format(self.network_domain_selector, self.location))
        return None
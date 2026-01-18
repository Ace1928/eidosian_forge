from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
class VlanDiff(object):
    """
    Represents differences between VLAN information (from CloudControl) and module parameters.
    """

    def __init__(self, vlan, module_params):
        """

        :param vlan: The VLAN information from CloudControl.
        :type vlan: DimensionDataVlan
        :param module_params: The module parameters.
        :type module_params: dict
        """
        self.vlan = vlan
        self.module_params = module_params
        self.name_changed = module_params['name'] != vlan.name
        self.description_changed = module_params['description'] != vlan.description
        self.private_ipv4_base_address_changed = module_params['private_ipv4_base_address'] != vlan.private_ipv4_range_address
        self.private_ipv4_prefix_size_changed = module_params['private_ipv4_prefix_size'] != vlan.private_ipv4_range_size
        private_ipv4_prefix_size_difference = module_params['private_ipv4_prefix_size'] - vlan.private_ipv4_range_size
        self.private_ipv4_prefix_size_increased = private_ipv4_prefix_size_difference > 0
        self.private_ipv4_prefix_size_decreased = private_ipv4_prefix_size_difference < 0

    def has_changes(self):
        """
        Does the VlanDiff represent any changes between the VLAN and module configuration?

        :return: True, if there are change changes; otherwise, False.
        """
        return self.needs_edit() or self.needs_expand()

    def ensure_legal_change(self):
        """
        Ensure the change (if any) represented by the VlanDiff represents a legal change to VLAN state.

        - private_ipv4_base_address cannot be changed
        - private_ipv4_prefix_size must be greater than or equal to the VLAN's existing private_ipv4_range_size

        :raise InvalidVlanChangeError: The VlanDiff does not represent a legal change to VLAN state.
        """
        if self.private_ipv4_base_address_changed:
            raise InvalidVlanChangeError('Cannot change the private IPV4 base address for an existing VLAN.')
        if self.private_ipv4_prefix_size_increased:
            raise InvalidVlanChangeError('Cannot shrink the private IPV4 network for an existing VLAN (only expand is supported).')

    def needs_edit(self):
        """
        Is an Edit operation required to resolve the differences between the VLAN information and the module parameters?

        :return: True, if an Edit operation is required; otherwise, False.
        """
        return self.name_changed or self.description_changed

    def needs_expand(self):
        """
        Is an Expand operation required to resolve the differences between the VLAN information and the module parameters?

        The VLAN's network is expanded by reducing the size of its network prefix.

        :return: True, if an Expand operation is required; otherwise, False.
        """
        return self.private_ipv4_prefix_size_decreased
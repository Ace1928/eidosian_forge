from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
class DimensionDataVlanModule(DimensionDataModule):
    """
    The dimensiondata_vlan module for Ansible.
    """

    def __init__(self):
        """
        Create a new Dimension Data VLAN module.
        """
        super(DimensionDataVlanModule, self).__init__(module=AnsibleModule(argument_spec=DimensionDataModule.argument_spec_with_wait(name=dict(required=True, type='str'), description=dict(default='', type='str'), network_domain=dict(required=True, type='str'), private_ipv4_base_address=dict(default='', type='str'), private_ipv4_prefix_size=dict(default=0, type='int'), allow_expand=dict(required=False, default=False, type='bool'), state=dict(default='present', choices=['present', 'absent', 'readonly'])), required_together=DimensionDataModule.required_together()))
        self.name = self.module.params['name']
        self.description = self.module.params['description']
        self.network_domain_selector = self.module.params['network_domain']
        self.private_ipv4_base_address = self.module.params['private_ipv4_base_address']
        self.private_ipv4_prefix_size = self.module.params['private_ipv4_prefix_size']
        self.state = self.module.params['state']
        self.allow_expand = self.module.params['allow_expand']
        if self.wait and self.state != 'present':
            self.module.fail_json(msg='The wait parameter is only supported when state is "present".')

    def state_present(self):
        """
        Ensure that the target VLAN is present.
        """
        network_domain = self._get_network_domain()
        vlan = self._get_vlan(network_domain)
        if not vlan:
            if self.module.check_mode:
                self.module.exit_json(msg='VLAN "{0}" is absent from network domain "{1}" (should be present).'.format(self.name, self.network_domain_selector), changed=True)
            vlan = self._create_vlan(network_domain)
            self.module.exit_json(msg='Created VLAN "{0}" in network domain "{1}".'.format(self.name, self.network_domain_selector), vlan=vlan_to_dict(vlan), changed=True)
        else:
            diff = VlanDiff(vlan, self.module.params)
            if not diff.has_changes():
                self.module.exit_json(msg='VLAN "{0}" is present in network domain "{1}" (no changes detected).'.format(self.name, self.network_domain_selector), vlan=vlan_to_dict(vlan), changed=False)
                return
            try:
                diff.ensure_legal_change()
            except InvalidVlanChangeError as invalid_vlan_change:
                self.module.fail_json(msg='Unable to update VLAN "{0}" in network domain "{1}": {2}'.format(self.name, self.network_domain_selector, invalid_vlan_change))
            if diff.needs_expand() and (not self.allow_expand):
                self.module.fail_json(msg='The configured private IPv4 network size ({0}-bit prefix) for '.format(self.private_ipv4_prefix_size) + 'the VLAN differs from its current network size ({0}-bit prefix) '.format(vlan.private_ipv4_range_size) + 'and needs to be expanded. Use allow_expand=true if this is what you want.')
            if self.module.check_mode:
                self.module.exit_json(msg='VLAN "{0}" is present in network domain "{1}" (changes detected).'.format(self.name, self.network_domain_selector), vlan=vlan_to_dict(vlan), changed=True)
            if diff.needs_edit():
                vlan.name = self.name
                vlan.description = self.description
                self.driver.ex_update_vlan(vlan)
            if diff.needs_expand():
                vlan.private_ipv4_range_size = self.private_ipv4_prefix_size
                self.driver.ex_expand_vlan(vlan)
            self.module.exit_json(msg='Updated VLAN "{0}" in network domain "{1}".'.format(self.name, self.network_domain_selector), vlan=vlan_to_dict(vlan), changed=True)

    def state_readonly(self):
        """
        Read the target VLAN's state.
        """
        network_domain = self._get_network_domain()
        vlan = self._get_vlan(network_domain)
        if vlan:
            self.module.exit_json(vlan=vlan_to_dict(vlan), changed=False)
        else:
            self.module.fail_json(msg='VLAN "{0}" does not exist in network domain "{1}".'.format(self.name, self.network_domain_selector))

    def state_absent(self):
        """
        Ensure that the target VLAN is not present.
        """
        network_domain = self._get_network_domain()
        vlan = self._get_vlan(network_domain)
        if not vlan:
            self.module.exit_json(msg='VLAN "{0}" is absent from network domain "{1}".'.format(self.name, self.network_domain_selector), changed=False)
            return
        if self.module.check_mode:
            self.module.exit_json(msg='VLAN "{0}" is present in network domain "{1}" (should be absent).'.format(self.name, self.network_domain_selector), vlan=vlan_to_dict(vlan), changed=True)
        self._delete_vlan(vlan)
        self.module.exit_json(msg='Deleted VLAN "{0}" from network domain "{1}".'.format(self.name, self.network_domain_selector), changed=True)

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

    def _create_vlan(self, network_domain):
        vlan = self.driver.ex_create_vlan(network_domain, self.name, self.private_ipv4_base_address, self.description, self.private_ipv4_prefix_size)
        if self.wait:
            vlan = self._wait_for_vlan_state(vlan.id, 'NORMAL')
        return vlan

    def _delete_vlan(self, vlan):
        try:
            self.driver.ex_delete_vlan(vlan)
            if self.wait:
                self._wait_for_vlan_state(vlan, 'NOT_FOUND')
        except DimensionDataAPIException as api_exception:
            self.module.fail_json(msg='Failed to delete VLAN "{0}" due to unexpected error from the CloudControl API: {1}'.format(vlan.id, api_exception.msg))

    def _wait_for_vlan_state(self, vlan, state_to_wait_for):
        network_domain = self._get_network_domain()
        wait_poll_interval = self.module.params['wait_poll_interval']
        wait_time = self.module.params['wait_time']
        try:
            return self.driver.connection.wait_for_state(state_to_wait_for, self.driver.ex_get_vlan, wait_poll_interval, wait_time, vlan)
        except DimensionDataAPIException as api_exception:
            if api_exception.code != 'RESOURCE_NOT_FOUND':
                raise
            return DimensionDataVlan(id=vlan.id, status='NOT_FOUND', name='', description='', private_ipv4_range_address='', private_ipv4_range_size=0, ipv4_gateway='', ipv6_range_address='', ipv6_range_size=0, ipv6_gateway='', location=self.location, network_domain=network_domain)

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
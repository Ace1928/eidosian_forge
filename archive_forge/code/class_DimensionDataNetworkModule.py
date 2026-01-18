from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import HAS_LIBCLOUD, DimensionDataModule
from ansible.module_utils.common.text.converters import to_native
class DimensionDataNetworkModule(DimensionDataModule):
    """
    The dimensiondata_network module for Ansible.
    """

    def __init__(self):
        """
        Create a new Dimension Data network module.
        """
        super(DimensionDataNetworkModule, self).__init__(module=AnsibleModule(argument_spec=DimensionDataModule.argument_spec_with_wait(name=dict(type='str', required=True), description=dict(type='str', required=False), service_plan=dict(default='ESSENTIALS', choices=['ADVANCED', 'ESSENTIALS']), state=dict(default='present', choices=['present', 'absent'])), required_together=DimensionDataModule.required_together()))
        self.name = self.module.params['name']
        self.description = self.module.params['description']
        self.service_plan = self.module.params['service_plan']
        self.state = self.module.params['state']

    def state_present(self):
        network = self._get_network()
        if network:
            self.module.exit_json(changed=False, msg='Network already exists', network=self._network_to_dict(network))
        network = self._create_network()
        self.module.exit_json(changed=True, msg='Created network "%s" in datacenter "%s".' % (self.name, self.location), network=self._network_to_dict(network))

    def state_absent(self):
        network = self._get_network()
        if not network:
            self.module.exit_json(changed=False, msg='Network "%s" does not exist' % self.name, network=self._network_to_dict(network))
        self._delete_network(network)

    def _get_network(self):
        if self.mcp_version == '1.0':
            networks = self.driver.list_networks(location=self.location)
        else:
            networks = self.driver.ex_list_network_domains(location=self.location)
        matched_network = [network for network in networks if network.name == self.name]
        if matched_network:
            return matched_network[0]
        return None

    def _network_to_dict(self, network):
        network_dict = dict(id=network.id, name=network.name, description=network.description)
        if isinstance(network.location, NodeLocation):
            network_dict['location'] = network.location.id
        else:
            network_dict['location'] = network.location
        if self.mcp_version == '1.0':
            network_dict['private_net'] = network.private_net
            network_dict['multicast'] = network.multicast
            network_dict['status'] = None
        else:
            network_dict['private_net'] = None
            network_dict['multicast'] = None
            network_dict['status'] = network.status
        return network_dict

    def _create_network(self):
        if self.mcp_version == '2.0' and 'service_plan' not in self.module.params:
            self.module.fail_json(msg='service_plan required when creating network and location is MCP 2.0')
        try:
            if self.mcp_version == '1.0':
                network = self.driver.ex_create_network(self.location, self.name, description=self.description)
            else:
                network = self.driver.ex_create_network_domain(self.location, self.name, self.module.params['service_plan'], description=self.description)
        except DimensionDataAPIException as e:
            self.module.fail_json(msg='Failed to create new network: %s' % to_native(e), exception=traceback.format_exc())
        if self.module.params['wait'] is True:
            network = self._wait_for_network_state(network.id, 'NORMAL')
        return network

    def _delete_network(self, network):
        try:
            if self.mcp_version == '1.0':
                deleted = self.driver.ex_delete_network(network)
            else:
                deleted = self.driver.ex_delete_network_domain(network)
            if deleted:
                self.module.exit_json(changed=True, msg='Deleted network with id %s' % network.id)
            self.module.fail_json('Unexpected failure deleting network with id %s' % network.id)
        except DimensionDataAPIException as e:
            self.module.fail_json(msg='Failed to delete network: %s' % to_native(e), exception=traceback.format_exc())

    def _wait_for_network_state(self, net_id, state_to_wait_for):
        try:
            return self.driver.connection.wait_for_state(state_to_wait_for, self.driver.ex_get_network_domain, self.module.params['wait_poll_interval'], self.module.params['wait_time'], net_id)
        except DimensionDataAPIException as e:
            self.module.fail_json(msg='Network did not reach % state in time: %s' % (state_to_wait_for, to_native(e)), exception=traceback.format_exc())
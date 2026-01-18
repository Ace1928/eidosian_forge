from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridHaGroup:
    """
    Create, modify and delete HA Group configurations for StorageGRID
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=False, type='str'), ha_group_id=dict(required=False, type='str'), description=dict(required=False, type='str'), gateway_cidr=dict(required=False, type='str'), virtual_ips=dict(required=False, type='list', elements='str'), interfaces=dict(required=False, type='list', elements='dict', options=dict(node=dict(required=False, type='str'), interface=dict(required=False, type='str')))))
        parameter_map = {'name': 'name', 'description': 'description', 'gateway_cidr': 'gatewayCidr', 'virtual_ips': 'virtualIps'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['name', 'gateway_cidr', 'virtual_ips', 'interfaces'])], required_one_of=[('name', 'ha_group_id')])
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        if self.parameters['state'] == 'present':
            for k in parameter_map.keys():
                if self.parameters.get(k) is not None:
                    self.data[parameter_map[k]] = self.parameters[k]
            if self.parameters.get('interfaces') is not None:
                self.data['interfaces'] = self.build_node_interface_list()

    def build_node_interface_list(self):
        node_interfaces = []
        api = 'api/v3/grid/node-health'
        nodes, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        for node_interface in self.parameters['interfaces']:
            node_dict = {}
            node = next((item for item in nodes['data'] if item['name'] == node_interface['node']), None)
            if node is not None:
                node_dict['nodeId'] = node['id']
                node_dict['interface'] = node_interface['interface']
                node_interfaces.append(node_dict)
            else:
                self.module.fail_json(msg="Node '%s' is invalid" % node_interface['node'])
        return node_interfaces

    def get_ha_group_id(self):
        api = 'api/v3/private/ha-groups'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return next((item['id'] for item in response.get('data') if item['name'] == self.parameters['name']), None)

    def get_ha_group(self, ha_group_id):
        api = 'api/v3/private/ha-groups/%s' % ha_group_id
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def create_ha_group(self):
        api = 'api/v3/private/ha-groups'
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_ha_group(self, ha_group_id):
        api = 'api/v3/private/ha-groups/%s' % ha_group_id
        dummy, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def update_ha_group(self, ha_group_id):
        api = 'api/v3/private/ha-groups/%s' % ha_group_id
        response, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        ha_group = None
        if self.parameters.get('ha_group_id'):
            ha_group = self.get_ha_group(self.parameters['ha_group_id'])
        else:
            ha_group_id = self.get_ha_group_id()
            if ha_group_id:
                ha_group = self.get_ha_group(ha_group_id)
        cd_action = self.na_helper.get_cd_action(ha_group, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(ha_group, self.data)
        result_message = ''
        resp_data = {}
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'delete':
                self.delete_ha_group(ha_group['id'])
                result_message = 'HA Group deleted'
            elif cd_action == 'create':
                resp_data = self.create_ha_group()
                result_message = 'HA Group created'
            elif modify:
                resp_data = self.update_ha_group(ha_group['id'])
                result_message = 'HA Group updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)
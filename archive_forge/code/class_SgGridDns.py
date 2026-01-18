from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridDns(object):
    """
    Create, modify and delete DNS entries for StorageGRID
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), dns_servers=dict(required=True, type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = self.parameters['dns_servers']

    def get_grid_dns(self):
        api = 'api/v3/grid/dns-servers'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def update_grid_dns(self):
        api = 'api/v3/grid/dns-servers'
        response, error = self.rest_api.put(api, self.data)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        grid_dns = self.get_grid_dns()
        cd_action = self.na_helper.get_cd_action(grid_dns, self.parameters['dns_servers'])
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            dns_diff = [i for i in self.data + grid_dns if i not in self.data or i not in grid_dns]
            if dns_diff:
                update = True
            if update:
                self.na_helper.changed = True
        result_message = ''
        resp_data = grid_dns
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            else:
                resp_data = self.update_grid_dns()
                result_message = 'Grid DNS updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)
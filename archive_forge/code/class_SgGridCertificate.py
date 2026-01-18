from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridCertificate:
    """
    Update StorageGRID certificates
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), type=dict(required=True, type='str', choices=['storage-api', 'management']), server_certificate=dict(required=False, type='str'), ca_bundle=dict(required=False, type='str'), private_key=dict(required=False, type='str', no_log=True)))
        parameter_map = {'server_certificate': 'serverCertificateEncoded', 'ca_bundle': 'caBundleEncoded', 'private_key': 'privateKeyEncoded'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['server_certificate', 'private_key'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.data = {}
        if self.parameters['state'] == 'present':
            for k in parameter_map.keys():
                if self.parameters.get(k) is not None:
                    self.data[parameter_map[k]] = self.parameters[k]
        self.module.fail_json

    def get_grid_certificate(self, cert_type):
        api = 'api/v3/grid/%s' % cert_type
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def update_grid_certificate(self, cert_type):
        api = 'api/v3/grid/%s/update' % cert_type
        response, error = self.rest_api.post(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def apply(self):
        """
        Perform pre-checks, call functions and exit
        """
        cert_type = ''
        cd_action = None
        if self.parameters.get('type') == 'storage-api':
            cert_type = 'storage-api-certificate'
        elif self.parameters.get('type') == 'management':
            cert_type = 'management-certificate'
        cert_data = self.get_grid_certificate(cert_type)
        if cert_data['serverCertificateEncoded'] is None and cert_data['caBundleEncoded'] is None:
            cd_action = self.na_helper.get_cd_action(None, self.parameters)
        else:
            cd_action = self.na_helper.get_cd_action(cert_data, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            if self.data.get('serverCertificateEncoded') is not None and self.data.get('privateKeyEncoded') is not None:
                for item in ['serverCertificateEncoded', 'caBundleEncoded']:
                    if self.data.get(item) != cert_data.get(item):
                        update = True
            if update:
                self.na_helper.changed = True
        result_message = ''
        resp_data = cert_data
        if self.na_helper.changed:
            if self.module.check_mode:
                pass
            elif cd_action == 'delete':
                self.update_grid_certificate(cert_type)
                resp_data = self.get_grid_certificate(cert_type)
                result_message = 'Grid %s removed' % cert_type
            else:
                self.update_grid_certificate(cert_type)
                resp_data = self.get_grid_certificate(cert_type)
                result_message = 'Grid %s updated' % cert_type
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)
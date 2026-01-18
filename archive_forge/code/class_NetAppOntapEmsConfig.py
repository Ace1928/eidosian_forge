from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapEmsConfig:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), mail_from=dict(required=False, type='str'), mail_server=dict(required=False, type='str'), proxy_url=dict(required=False, type='str'), proxy_user=dict(required=False, type='str'), proxy_password=dict(required=False, type='str', no_log=True), pubsub_enabled=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=False)
        self.uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_ems_config:', 9, 6)
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, None, [['pubsub_enabled', (9, 10, 1)]])

    def get_ems_config_rest(self):
        """Get EMS config details"""
        fields = 'mail_from,mail_server,proxy_url,proxy_user'
        if 'pubsub_enabled' in self.parameters and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            fields += ',pubsub_enabled'
        record, error = rest_generic.get_one_record(self.rest_api, 'support/ems', None, fields)
        if error:
            self.module.fail_json(msg='Error fetching EMS config: %s' % to_native(error), exception=traceback.format_exc())
        if record:
            return {'mail_from': record.get('mail_from'), 'mail_server': record.get('mail_server'), 'proxy_url': record.get('proxy_url'), 'proxy_user': record.get('proxy_user'), 'pubsub_enabled': record.get('pubsub_enabled')}
        return None

    def modify_ems_config_rest(self, modify):
        """Modify EMS config"""
        dummy, error = rest_generic.patch_async(self.rest_api, 'support/ems', None, modify)
        if error:
            self.module.fail_json(msg='Error modifying EMS config: %s.' % to_native(error), exception=traceback.format_exc())

    def check_proxy_url(self, current):
        port = None
        if current.get('proxy_url') is not None:
            port = current['proxy_url'].rstrip('/').split(':')[-1]
        pos = self.parameters['proxy_url'].rstrip('/').rfind(':')
        if self.parameters['proxy_url'][pos + 1] == '/':
            if port is not None and port != '':
                self.parameters['proxy_url'] = '%s:%s' % (self.parameters['proxy_url'].rstrip('/'), port)

    def apply(self):
        current = self.get_ems_config_rest()
        if self.parameters.get('proxy_url') not in [None, '']:
            self.check_proxy_url(current)
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        password_changed = False
        if self.parameters.get('proxy_password') not in [None, '']:
            modify['proxy_password'] = self.parameters['proxy_password']
            self.module.warn('Module is not idempotent when proxy_password is set.')
            password_changed = True
        if (self.na_helper.changed or password_changed) and (not self.module.check_mode):
            self.modify_ems_config_rest(modify)
        result = netapp_utils.generate_result(changed=self.na_helper.changed | password_changed, modify=modify)
        self.module.exit_json(**result)
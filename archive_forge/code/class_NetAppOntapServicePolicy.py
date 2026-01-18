from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapServicePolicy:
    """
    Common operations to manage public keys.
    """

    def __init__(self):
        self.use_rest = False
        argument_spec = netapp_utils.na_ontap_host_argument_spec()
        argument_spec.update(dict(state=dict(type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), ipspace=dict(type='str'), scope=dict(type='str', choices=['cluster', 'svm']), services=dict(type='list', elements='str'), vserver=dict(type='str'), known_services=dict(type='list', elements='str', default=['cluster_core', 'intercluster_core', 'management_core', 'management_autosupport', 'management_bgp', 'management_ems', 'management_https', 'management_http', 'management_ssh', 'management_portmap', 'data_core', 'data_nfs', 'data_cifs', 'data_flexcache', 'data_iscsi', 'data_s3_server', 'data_dns_server', 'data_fpolicy_client', 'management_ntp_client', 'management_dns_client', 'management_ad_client', 'management_ldap_client', 'management_nis_client', 'management_snmp_server', 'management_rsh_server', 'management_telnet_server', 'management_ntp_server', 'data_nvme_tcp', 'backup_ndmp_control', 'management_log_forwarding']), additional_services=dict(type='list', elements='str')))
        self.module = AnsibleModule(argument_spec=argument_spec, required_if=[('scope', 'cluster', ['ipspace']), ('scope', 'svm', ['vserver']), ('vserver', None, ['ipspace'])], required_one_of=[('ipspace', 'vserver')], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_service_policy', 9, 8)
        self.validate_inputs()

    def validate_inputs(self):
        services = self.parameters.get('services')
        if services and 'no_service' in services:
            if len(services) > 1:
                self.module.fail_json(msg='Error: no other service can be present when no_service is specified.  Got: %s' % services)
            self.parameters['services'] = []
        known_services = self.parameters.get('known_services', []) + self.parameters.get('additional_services', [])
        unknown_services = [service for service in self.parameters.get('services', []) if service not in known_services]
        if unknown_services:
            plural = 's' if len(services) > 1 else ''
            self.module.fail_json(msg='Error: unknown service%s: %s.  New services may need to be added to "additional_services".' % (plural, ','.join(unknown_services)))
        scope = self.parameters.get('scope')
        if scope is None:
            self.parameters['scope'] = 'cluster' if self.parameters.get('vserver') is None else 'svm'
        elif scope == 'cluster' and self.parameters.get('vserver') is not None:
            self.module.fail_json(msg='Error: vserver cannot be set when "scope: cluster" is specified.  Got: %s' % self.parameters.get('vserver'))
        elif scope == 'svm' and self.parameters.get('vserver') is None:
            self.module.fail_json(msg='Error: vserver cannot be None when "scope: svm" is specified.')

    def get_service_policy(self):
        api = 'network/ip/service-policies'
        query = {'name': self.parameters['name'], 'fields': 'name,uuid,ipspace,services,svm,scope'}
        if self.parameters.get('vserver') is None:
            query['scope'] = 'cluster'
        else:
            query['svm.name'] = self.parameters['vserver']
        if self.parameters.get('ipspace') is not None:
            query['ipspace.name'] = self.parameters['ipspace']
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            msg = 'Error in get_service_policy: %s' % error
            self.module.fail_json(msg=msg)
        if record:
            return {'uuid': record['uuid'], 'name': record['name'], 'ipspace': record['ipspace']['name'], 'scope': record['scope'], 'vserver': self.na_helper.safe_get(record, ['svm', 'name']), 'services': record['services']}
        return None

    def create_service_policy(self):
        api = 'network/ip/service-policies'
        body = {'name': self.parameters['name']}
        if self.parameters.get('vserver') is not None:
            body['svm.name'] = self.parameters['vserver']
        for attr in ('ipspace', 'scope', 'services'):
            value = self.parameters.get(attr)
            if value is not None:
                body[attr] = value
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            msg = 'Error in create_service_policy: %s' % error
            self.module.fail_json(msg=msg)

    def modify_service_policy(self, current, modify):
        api = 'network/ip/service-policies/%s' % current['uuid']
        modify_copy = dict(modify)
        body = {}
        for key in modify:
            if key in ('services',):
                body[key] = modify_copy.pop(key)
        if modify_copy:
            msg = 'Error: attributes not supported in modify: %s' % modify_copy
            self.module.fail_json(msg=msg)
        if not body:
            msg = 'Error: nothing to change - modify called with: %s' % modify
            self.module.fail_json(msg=msg)
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
        if error:
            msg = 'Error in modify_service_policy: %s' % error
            self.module.fail_json(msg=msg)

    def delete_service_policy(self, current):
        api = 'network/ip/service-policies/%s' % current['uuid']
        dummy, error = rest_generic.delete_async(self.rest_api, api, None, None)
        if error:
            msg = 'Error in delete_service_policy: %s' % error
            self.module.fail_json(msg=msg)

    def get_actions(self):
        """Determines whether a create, delete, modify action is required
        """
        cd_action, modify, current = (None, None, None)
        current = self.get_service_policy()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        return (cd_action, modify, current)

    def apply(self):
        cd_action, modify, current = self.get_actions()
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_service_policy()
            elif cd_action == 'delete':
                self.delete_service_policy(current)
            elif modify:
                self.modify_service_policy(current, modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify, extra_responses={'scope': self.module.params})
        self.module.exit_json(**result)
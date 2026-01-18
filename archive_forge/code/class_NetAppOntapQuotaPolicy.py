from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import zapis_svm
class NetAppOntapQuotaPolicy(object):
    """
        Create, assign, rename or delete a quota policy
    """

    def __init__(self):
        """
            Initialize the ONTAP quota policy class
        """
        self.argument_spec = netapp_utils.na_ontap_zapi_only_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), name=dict(required=True, type='str'), from_name=dict(required=False, type='str'), auto_assign=dict(required=False, type='bool', default=True)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['name', 'vserver'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.na_helper.module_deprecated(self.module)
        if HAS_NETAPP_LIB is False:
            self.module.fail_json(msg='The python NetApp-Lib module is required')
        else:
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])

    def get_quota_policy(self, policy_name=None):
        if policy_name is None:
            policy_name = self.parameters['name']
        return_value = None
        quota_policy_get_iter = netapp_utils.zapi.NaElement('quota-policy-get-iter')
        quota_policy_info = netapp_utils.zapi.NaElement('quota-policy-info')
        quota_policy_info.add_new_child('policy-name', policy_name)
        quota_policy_info.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(quota_policy_info)
        quota_policy_get_iter.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(quota_policy_get_iter, True)
            if result.get_child_by_name('attributes-list'):
                quota_policy_attributes = result['attributes-list']['quota-policy-info']
                return_value = {'name': quota_policy_attributes['policy-name']}
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching quota policy %s: %s' % (policy_name, to_native(error)), exception=traceback.format_exc())
        return return_value

    def create_quota_policy(self):
        """
        Creates a new quota policy
        """
        quota_policy_obj = netapp_utils.zapi.NaElement('quota-policy-create')
        quota_policy_obj.add_new_child('policy-name', self.parameters['name'])
        quota_policy_obj.add_new_child('vserver', self.parameters['vserver'])
        try:
            self.server.invoke_successfully(quota_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error creating quota policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_quota_policy(self):
        """
        Deletes a quota policy
        """
        quota_policy_obj = netapp_utils.zapi.NaElement('quota-policy-delete')
        quota_policy_obj.add_new_child('policy-name', self.parameters['name'])
        try:
            self.server.invoke_successfully(quota_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting quota policy %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def rename_quota_policy(self):
        """
        Rename a quota policy
        """
        quota_policy_obj = netapp_utils.zapi.NaElement('quota-policy-rename')
        quota_policy_obj.add_new_child('policy-name', self.parameters['from_name'])
        quota_policy_obj.add_new_child('vserver', self.parameters['vserver'])
        quota_policy_obj.add_new_child('new-policy-name', self.parameters['name'])
        try:
            self.server.invoke_successfully(quota_policy_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming quota policy %s: %s' % (self.parameters['from_name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        current = self.get_quota_policy()
        rename, cd_action = (None, None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and self.parameters.get('from_name'):
            rename = self.na_helper.is_rename_action(self.get_quota_policy(self.parameters['from_name']), current)
            if rename is None:
                self.module.fail_json(msg='Error renaming quota policy: %s does not exist.' % self.parameters['from_name'])
        assign_policy = cd_action == 'create' and self.parameters['auto_assign']
        if cd_action is None and current and self.parameters['auto_assign']:
            svm = zapis_svm.get_vserver(self.server, self.parameters['vserver'])
            if svm.get('quota_policy') != self.parameters['name']:
                assign_policy = True
                self.na_helper.changed = True
        if cd_action == 'delete':
            svm = zapis_svm.get_vserver(self.server, self.parameters['vserver'])
            if svm.get('quota_policy') == self.parameters['name']:
                self.module.fail_json(msg='Error policy %s cannot be deleted as it is assigned to the vserver %s' % (self.parameters['name'], self.parameters['vserver']))
        if self.na_helper.changed and (not self.module.check_mode):
            if rename:
                self.rename_quota_policy()
            elif cd_action == 'create':
                self.create_quota_policy()
            elif cd_action == 'delete':
                self.delete_quota_policy()
            if assign_policy:
                zapis_svm.modify_vserver(self.server, self.module, self.parameters['vserver'], modify=dict(quota_policy=self.parameters['name']))
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)
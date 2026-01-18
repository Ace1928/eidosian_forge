from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapCifsLocalGroup:
    """
        Create, delete or modify CIFS local group
    """

    def __init__(self):
        """
            Initialize the Ontap CifsLocalGroup class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), name=dict(required=True, type='str'), description=dict(required=False, type='str'), from_name=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_cifs_local_group', 9, 10, 1)
        self.svm_uuid = None
        self.sid = None

    def get_cifs_local_group_rest(self, from_name=None):
        """
        Retrieves the local group of an SVM.
        """
        api = 'protocols/cifs/local-groups'
        query = {'name': from_name or self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,sid,description'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching cifs local-group: %s' % error)
        if record:
            self.svm_uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
            self.sid = self.na_helper.safe_get(record, ['sid'])
            return {'name': self.na_helper.safe_get(record, ['name']), 'description': record.get('description', '')}
        return None

    def create_cifs_local_group_rest(self):
        """
        Creates the local group of an SVM.
        """
        api = 'protocols/cifs/local-groups'
        body = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver']}
        if 'description' in self.parameters:
            body['description'] = self.parameters['description']
        record, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error on creating cifs local-group: %s' % error)

    def delete_cifs_local_group_rest(self):
        """
        Destroy the local group of an SVM.
        """
        api = 'protocols/cifs/local-groups/%s/%s' % (self.svm_uuid, self.sid)
        record, error = rest_generic.delete_async(self.rest_api, api, None)
        if error:
            self.module.fail_json(msg='Error on deleting cifs local-group: %s' % error)

    def modify_cifs_local_group_rest(self, modify):
        """
        Modify the description of CIFS local group.
        Rename cifs local group.
        """
        body = {}
        if 'description' in modify:
            body['description'] = self.parameters['description']
        if 'name' in modify:
            body['name'] = self.parameters['name']
        api = 'protocols/cifs/local-groups/%s/%s' % (self.svm_uuid, self.sid)
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
        if error is not None:
            self.module.fail_json(msg='Error on modifying cifs local-group: %s' % error)

    def apply(self):
        current = self.get_cifs_local_group_rest()
        rename = None
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and 'from_name' in self.parameters:
            group_info = self.get_cifs_local_group_rest(self.parameters['from_name'])
            rename = self.na_helper.is_rename_action(group_info, current)
            if rename:
                current = group_info
                cd_action = None
            else:
                self.module.fail_json(msg='Error renaming cifs local group: %s - no cifs local group with from_name: %s.' % (self.parameters['name'], self.parameters['from_name']))
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_cifs_local_group_rest()
            elif cd_action == 'delete':
                self.delete_cifs_local_group_rest()
            if modify or rename:
                self.modify_cifs_local_group_rest(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)
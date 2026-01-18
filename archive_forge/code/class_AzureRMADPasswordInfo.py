from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMADPasswordInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(app_id=dict(type='str'), app_object_id=dict(type='str'), service_principal_object_id=dict(type='str'), key_id=dict(type='str'), tenant=dict(type='str', required=True), value=dict(type='str'), end_date=dict(type='str'))
        self.tenant = None
        self.app_id = None
        self.service_principal_object_id = None
        self.app_object_id = None
        self.key_id = None
        self.value = None
        self.end_date = None
        self.results = dict(changed=False)
        self.client = None
        super(AzureRMADPasswordInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_tags=False, supports_check_mode=True, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self.client = self.get_graphrbac_client(self.tenant)
        self.resolve_app_obj_id()
        passwords = self.get_all_passwords()
        if self.key_id:
            filtered = [pd for pd in passwords if pd.key_id == self.key_id]
            self.results['passwords'] = [self.to_dict(pd) for pd in filtered]
        else:
            self.results['passwords'] = [self.to_dict(pd) for pd in passwords]
        return self.results

    def resolve_app_obj_id(self):
        try:
            if self.app_object_id is not None:
                return
            elif self.app_id or self.service_principal_object_id:
                if not self.app_id:
                    sp = self.client.service_principals.get(self.service_principal_id)
                    self.app_id = sp.app_id
                if not self.app_id:
                    self.fail("can't resolve app via service principal object id {0}".format(self.service_principal_object_id))
                result = list(self.client.applications.list(filter="appId eq '{0}'".format(self.app_id)))
                if result:
                    self.app_object_id = result[0].object_id
                else:
                    self.fail("can't resolve app via app id {0}".format(self.app_id))
            else:
                self.fail('one of the [app_id, app_object_id, service_principal_id] must be set')
        except GraphErrorException as ge:
            self.fail('error in resolve app_object_id {0}'.format(str(ge)))

    def get_all_passwords(self):
        try:
            return list(self.client.applications.list_password_credentials(self.app_object_id))
        except GraphErrorException as ge:
            self.fail('failed to fetch passwords for app {0}: {1}'.format(self.app_object_id, str(ge)))

    def to_dict(self, pd):
        return dict(end_date=pd.end_date, start_date=pd.start_date, key_id=pd.key_id, custom_key_identifier=str(pd.custom_key_identifier))
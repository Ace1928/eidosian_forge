from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMADUserInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(user_principal_name=dict(type='str'), object_id=dict(type='str'), attribute_name=dict(type='str'), attribute_value=dict(type='str'), odata_filter=dict(type='str'), all=dict(type='bool'), tenant=dict(type='str', required=True))
        self.tenant = None
        self.user_principal_name = None
        self.object_id = None
        self.attribute_name = None
        self.attribute_value = None
        self.odata_filter = None
        self.all = None
        self.log_path = None
        self.log_mode = None
        self.results = dict(changed=False)
        mutually_exclusive = [['odata_filter', 'attribute_name', 'object_id', 'user_principal_name', 'all']]
        required_together = [['attribute_name', 'attribute_value']]
        required_one_of = [['odata_filter', 'attribute_name', 'object_id', 'user_principal_name', 'all']]
        super(AzureRMADUserInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, mutually_exclusive=mutually_exclusive, required_together=required_together, required_one_of=required_one_of, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        ad_users = []
        try:
            client = self.get_graphrbac_client(self.tenant)
            if self.user_principal_name is not None:
                ad_users = [client.users.get(self.user_principal_name)]
            elif self.object_id is not None:
                ad_users = [client.users.get(self.object_id)]
            elif self.attribute_name is not None and self.attribute_value is not None:
                try:
                    ad_users = list(client.users.list(filter="{0} eq '{1}'".format(self.attribute_name, self.attribute_value)))
                except GraphErrorException as e:
                    try:
                        ad_users = list(client.users.list(filter="{0}/any(c:c eq '{1}')".format(self.attribute_name, self.attribute_value)))
                    except GraphErrorException as sub_e:
                        raise
            elif self.odata_filter is not None:
                ad_users = list(client.users.list(filter=self.odata_filter))
            elif self.all:
                ad_users = list(client.users.list())
            self.results['ad_users'] = [self.to_dict(user) for user in ad_users]
        except GraphErrorException as e:
            self.fail('failed to get ad user info {0}'.format(str(e)))
        return self.results

    def to_dict(self, object):
        return dict(object_id=object.object_id, display_name=object.display_name, user_principal_name=object.user_principal_name, mail_nickname=object.mail_nickname, mail=object.mail, account_enabled=object.account_enabled, user_type=object.user_type)
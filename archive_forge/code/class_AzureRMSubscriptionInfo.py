from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSubscriptionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', aliases=['subscription_name']), id=dict(type='str'), tags=dict(type='list', elements='str'), all=dict(type='bool'))
        self.results = dict(changed=False, subscriptions=[])
        self.name = None
        self.id = None
        self.tags = None
        self.all = False
        mutually_exclusive = [['name', 'id']]
        super(AzureRMSubscriptionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, mutually_exclusive=mutually_exclusive, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.id and self.name:
            self.fail('Parameter error: cannot search subscriptions by both name and id.')
        result = []
        if self.id:
            result = self.get_item()
        else:
            result = self.list_items()
        self.results['subscriptions'] = result
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.id))
        item = None
        result = []
        try:
            item = self.subscription_client.subscriptions.get(self.id)
        except ResourceNotFoundError:
            pass
        result = self.to_dict(item)
        return result

    def list_items(self):
        self.log('List all items')
        try:
            response = self.subscription_client.subscriptions.list()
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.name and self.name.lower() == item.display_name.lower():
                results.append(self.to_dict(item))
            elif not self.name and (self.all or item.state == 'Enabled'):
                results.append(self.to_dict(item))
        return results

    def to_dict(self, subscription_object):
        if self.has_tags(subscription_object.tags, self.tags):
            return dict(display_name=subscription_object.display_name, fqid=subscription_object.id, state=subscription_object.state, subscription_id=subscription_object.subscription_id, tags=subscription_object.tags, tenant_id=subscription_object.tenant_id)
        else:
            return dict()
from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMManagementGroupInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(children=dict(type='bool', default=False), flatten=dict(type='bool', default=False), id=dict(type='str'), name=dict(type='str', aliases=['management_group_name']), recurse=dict(type='bool', default=False))
        self.results = dict(changed=False, management_groups=[])
        self.children = None
        self.flatten = None
        self.id = None
        self.name = None
        self.recurse = None
        mutually_exclusive = [['name', 'id']]
        super(AzureRMManagementGroupInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, mutually_exclusive=mutually_exclusive, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        response = []
        if self.name or self.id:
            response = [self.get_item()]
        else:
            response = self.list_items()
        if self.flatten and self.children:
            self.results['subscriptions'] = []
            for group in response:
                new_groups = []
                new_subscriptions = []
                self.results['management_groups'].append(group)
                new_groups, new_subscriptions = self.flatten_group(group)
                self.results['management_groups'] += new_groups
                self.results['subscriptions'] += new_subscriptions
        else:
            self.results['management_groups'] = response
        return self.results

    def get_item(self, mg_name=None):
        if not mg_name:
            if self.id and (not self.name):
                mg_name = self.id.split('/')[-1]
            else:
                mg_name = self.name
        expand = 'children' if self.children else None
        try:
            response = self.management_groups_client.management_groups.get(group_id=mg_name, expand=expand, recurse=self.recurse)
        except Exception as e:
            self.log('No Management group {0} found. msg: {1}'.format(mg_name, e))
            response = []
        return self.to_dict(response)

    def list_items(self):
        self.log('List all management groups.')
        results = []
        response = []
        try:
            response = self.management_groups_client.management_groups.list()
        except Exception as e:
            self.log('No Management groups found.msg: {0}'.format(e))
            pass
        if self.children:
            results = [self.get_item(mg_name=item.name) for item in response]
        else:
            results = [self.to_dict(item) for item in response]
        return results

    def to_dict(self, azure_object):
        if not azure_object:
            return []
        if azure_object.type == 'Microsoft.Management/managementGroups':
            return_dict = dict(display_name=azure_object.display_name, id=azure_object.id, name=azure_object.name, type=azure_object.type)
            if self.children and azure_object.as_dict().get('children'):
                return_dict['children'] = [self.to_dict(item) for item in azure_object.children]
            elif self.children:
                return_dict['children'] = []
            if azure_object.as_dict().get('details', {}).get('parent'):
                parent_dict = azure_object.as_dict().get('details', {}).get('parent')
                return_dict['parent'] = dict(display_name=parent_dict.get('display_name'), id=parent_dict.get('id'), name=parent_dict.get('name'))
        elif azure_object.type == '/subscriptions':
            return_dict = dict(display_name=azure_object.display_name, id=azure_object.id, subscription_id=azure_object.name, type=azure_object.type)
        else:
            return_dict = dict(state='This is an unknown and unexpected object. ' + 'You should report this as a bug to the ansible-collection/azcollection ' + 'project on github. Please include the object type in your issue report, ' + 'and @ the authors of this module. ', type=azure_object.as_dict().get('type', None))
        if azure_object.as_dict().get('tenant_id'):
            return_dict['tenant_id'] = azure_object.tenant_id
        return return_dict

    def flatten_group(self, management_group):
        management_group_list = []
        subscription_list = []
        if management_group.get('children'):
            for child in management_group.get('children', []):
                if child.get('type') == '/providers/Microsoft.Management/managementGroups':
                    management_group_list.append(child)
                    new_groups, new_subscriptions = self.flatten_group(child)
                    management_group_list += new_groups
                    subscription_list += new_subscriptions
                elif child.get('type') == '/subscriptions':
                    subscription_list.append(child)
        return (management_group_list, subscription_list)
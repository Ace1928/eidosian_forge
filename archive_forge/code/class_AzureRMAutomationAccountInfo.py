from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAutomationAccountInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'), list_statistics=dict(type='bool'), list_usages=dict(type='bool'), list_keys=dict(type='bool'))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.tags = None
        self.list_statistics = None
        self.list_usages = None
        self.list_keys = None
        super(AzureRMAutomationAccountInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_automationaccount_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_automationaccount_facts' module has been renamed to 'azure_rm_automationaccount_info'", version=(2.9,))
        for key in list(self.module_arg_spec):
            setattr(self, key, kwargs[key])
        if self.resource_group and self.name:
            accounts = [self.get()]
        elif self.resource_group:
            accounts = self.list_by_resource_group()
        else:
            accounts = self.list_all()
        self.results['automation_accounts'] = [self.to_dict(x) for x in accounts if self.has_tags(x.tags, self.tags)]
        return self.results

    def to_dict(self, account):
        if not account:
            return None
        id_dict = parse_resource_id(account.id)
        result = account.as_dict()
        result['resource_group'] = id_dict['resource_group']
        if self.list_statistics:
            result['statistics'] = self.get_statics(id_dict['resource_group'], account.name)
        if self.list_usages:
            result['usages'] = self.get_usages(id_dict['resource_group'], account.name)
        if self.list_keys:
            result['keys'] = self.list_account_keys(id_dict['resource_group'], account.name)
        return result

    def get(self):
        try:
            return self.automation_client.automation_account.get(self.resource_group, self.name)
        except ResourceNotFoundError as exc:
            self.fail('Error when getting automation account {0}: {1}'.format(self.name, exc.message))

    def list_by_resource_group(self):
        result = []
        try:
            resp = self.automation_client.automation_account.list_by_resource_group(self.resource_group)
            while True:
                result.append(resp.next())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when listing automation account in resource group {0}: {1}'.format(self.resource_group, exc.message))
        return result

    def list_all(self):
        result = []
        try:
            resp = self.automation_client.automation_account.list()
            while True:
                result.append(resp.next())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when listing automation account: {0}'.format(exc.message))
        return result

    def get_statics(self, resource_group, name):
        result = []
        try:
            resp = self.automation_client.statistics.list_by_automation_account(resource_group, name)
            while True:
                result.append(resp.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when getting statics for automation account {0}/{1}: {2}'.format(resource_group, name, exc.message))
        return result

    def get_usages(self, resource_group, name):
        result = []
        try:
            resp = self.automation_client.usages.list_by_automation_account(resource_group, name)
            while True:
                result.append(resp.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when getting usage for automation account {0}/{1}: {2}'.format(resource_group, name, exc.message))
        return result

    def list_account_keys(self, resource_group, name):
        try:
            resp = self.automation_client.keys.list_by_automation_account(resource_group, name)
            return [x.as_dict() for x in resp.keys]
        except Exception as exc:
            self.fail('Error when listing keys for automation account {0}/{1}: {2}'.format(resource_group, name, exc.message))
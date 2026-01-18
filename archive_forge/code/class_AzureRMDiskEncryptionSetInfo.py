from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDiskEncryptionSetInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMDiskEncryptionSetInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        results = []
        if self.name is not None:
            results = self.get_item()
        elif self.resource_group:
            results = self.list_resource_group()
        else:
            results = self.list_items()
        self.results['diskencryptionsets'] = self.curated_items(results)
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.compute_client.disk_encryption_sets.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [item]
        return results

    def list_resource_group(self):
        self.log('List all disk encryption sets for resource group - {0}'.format(self.resource_group))
        try:
            response = self.compute_client.disk_encryption_sets.list_by_resource_group(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def list_items(self):
        self.log('List all disk encryption sets for a subscription ')
        try:
            response = self.compute_client.disk_encryption_sets.list()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def curated_items(self, raws):
        return [self.diskencryptionset_to_dict(item) for item in raws] if raws else []

    def diskencryptionset_to_dict(self, diskencryptionset):
        result = dict(id=diskencryptionset.id, name=diskencryptionset.name, location=diskencryptionset.location, tags=diskencryptionset.tags, active_key=diskencryptionset.active_key.as_dict(), provisioning_state=diskencryptionset.provisioning_state, identity=diskencryptionset.identity.as_dict(), type=diskencryptionset.type)
        return result
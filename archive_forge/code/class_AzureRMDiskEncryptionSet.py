from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, \
class AzureRMDiskEncryptionSet(AzureRMModuleBase):

    def __init__(self):
        _load_params()
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), source_vault=dict(type='str'), key_url=dict(type='str', no_log=True), state=dict(choices=['present', 'absent'], default='present', type='str'))
        required_if = [('state', 'present', ['source_vault', 'key_url'])]
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.location = None
        self.source_vault = None
        self.key_url = None
        self.state = None
        self.tags = None
        super(AzureRMDiskEncryptionSet, self).__init__(self.module_arg_spec, required_if=required_if, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        results = dict()
        disk_encryption_set = None
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        if self.source_vault:
            source_vault = self.parse_resource_to_dict(self.source_vault)
            self.source_vault = format_resource_id(val=source_vault['name'], subscription_id=source_vault['subscription_id'], namespace='Microsoft.KeyVault', types='vaults', resource_group=source_vault['resource_group'])
        try:
            self.log('Fetching Disk encryption set {0}'.format(self.name))
            disk_encryption_set_old = self.compute_client.disk_encryption_sets.get(self.resource_group, self.name)
            results = self.diskencryptionset_to_dict(disk_encryption_set_old)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                self.tags = results['tags']
                if self.source_vault != results['active_key']['source_vault']['id']:
                    changed = True
                    results['active_key']['source_vault']['id'] = self.source_vault
                if self.key_url != results['active_key']['key_url']:
                    changed = True
                    results['active_key']['key_url'] = self.key_url
            elif self.state == 'absent':
                changed = True
        except ResourceNotFoundError:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                identity = self.compute_models.EncryptionSetIdentity(type='SystemAssigned')
                disk_encryption_set_new = self.compute_models.DiskEncryptionSet(location=self.location, identity=identity)
                if self.source_vault:
                    source_vault = self.compute_models.SourceVault(id=self.source_vault)
                    disk_encryption_set_new.active_key = self.compute_models.KeyVaultAndKeyReference(source_vault=source_vault, key_url=self.key_url)
                if self.tags:
                    disk_encryption_set_new.tags = self.tags
                self.results['state'] = self.create_or_update_diskencryptionset(disk_encryption_set_new)
            elif self.state == 'absent':
                self.delete_diskencryptionset()
                self.results['state'] = 'Deleted'
        return self.results

    def create_or_update_diskencryptionset(self, disk_encryption_set):
        try:
            response = self.compute_client.disk_encryption_sets.begin_create_or_update(resource_group_name=self.resource_group, disk_encryption_set_name=self.name, disk_encryption_set=disk_encryption_set)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error creating or updating disk encryption set {0} - {1}'.format(self.name, str(exc)))
        return self.diskencryptionset_to_dict(response)

    def delete_diskencryptionset(self):
        try:
            response = self.compute_client.disk_encryption_sets.begin_delete(resource_group_name=self.resource_group, disk_encryption_set_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error deleting disk encryption set {0} - {1}'.format(self.name, str(exc)))
        return response

    def diskencryptionset_to_dict(self, diskencryptionset):
        result = diskencryptionset.as_dict()
        result['tags'] = diskencryptionset.tags
        return result
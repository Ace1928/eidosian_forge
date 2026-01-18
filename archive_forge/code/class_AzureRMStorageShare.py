from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMStorageShare(AzureRMModuleBase):
    """
    Configuration class for an Azure RM Storage file share resource
    """

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), account_name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), access_tier=dict(type='str', default=None, choices=['TransactionOptimized', 'Hot', 'Cool', 'Premium']), quota=dict(type='int', default=None), metadata=dict(type='dict', default=None), root_squash=dict(type='str', choices=['NoRootSquash', 'RootSquash', 'AllSquash']), enabled_protocols=dict(type='str', choices=['SMB', 'NFS']))
        self.results = dict(changed=False, state=dict())
        self.resource_group = None
        self.name = None
        self.account_name = None
        self.state = None
        self.quota = None
        self.metadata = None
        self.root_squash = None
        self.enabled_protocols = None
        self.to_do = Actions.NoAction
        super(AzureRMStorageShare, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        """
        Main module execution method
        """
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self.log('Fetching storage file share {0}'.format(self.name))
        response = None
        old_response = self.get_share()
        if old_response is None:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            self.to_do = Actions.Update
        if self.to_do == Actions.Create:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_storage_share()
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.delete_storage_share()
        elif self.to_do == Actions.Update:
            if self.update_needed(old_response):
                self.results['changed'] = True
                if self.check_mode:
                    return self.results
                response = self.update_storage_share(old_response)
            else:
                self.results['changed'] = False
                response = old_response
        if response is not None:
            self.results['state'] = response
        else:
            self.results['state'] = dict()
        return self.results

    def update_needed(self, old_response):
        """
        Define if storage file share update needed.
        :param old_response: dict with properties of the storage file share
        :return: True if update needed, else False
        """
        return self.access_tier is not None and self.access_tier != old_response.get('access_tier') or (self.quota is not None and self.quota != old_response.get('share_quota')) or (self.metadata is not None and self.metadata != old_response.get('metadata')) or (self.root_squash is not None and self.root_squash != old_response.get('root_squash')) or (self.enabled_protocols is not None and self.enabled_protocols != old_response.get('enabled_protocols'))

    def get_share(self):
        """
        Get the properties of the specified Azure Storage file share.
        :return: dict with properties of the storage file share
        """
        found = False
        try:
            storage_share = self.storage_client.file_shares.get(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name)
            found = True
            self.log('Response : {0}'.format(storage_share))
        except ResourceNotFoundError as e:
            self.log('Did not find the storage file share with name {0} : {1}'.format(self.name, str(e)))
        return self.storage_share_to_dict(storage_share) if found else None

    def storage_share_to_dict(self, storage_share):
        """
        Transform Azure RM Storage share object to dictionary
        :param storage_share: contains information about storage file share
        :type storage_share: FileShare
        :return: dict generated from storage_share
        """
        return dict(id=storage_share.id, name=storage_share.name, type=storage_share.type, etag=storage_share.etag.replace('"', ''), last_modified_time=storage_share.last_modified_time, metadata=storage_share.metadata, share_quota=storage_share.share_quota, access_tier=storage_share.access_tier, access_tier_change_time=storage_share.access_tier_change_time, root_squash=storage_share.root_squash, enabled_protocols=storage_share.enabled_protocols)

    def create_storage_share(self):
        """
        Method calling the Azure SDK to create storage file share.
        :return: dict with description of the new storage file share
        """
        self.log('Creating fileshare {0}'.format(self.name))
        try:
            self.storage_client.file_shares.create(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name, file_share=dict(access_tier=self.access_tier, share_quota=self.quota, metadata=self.metadata, root_squash=self.root_squash, enabled_protocols=self.enabled_protocols))
        except Exception as e:
            self.fail('Error creating file share {0} : {1}'.format(self.name, str(e)))
        return self.get_share()

    def update_storage_share(self, old_responce):
        """
        Method calling the Azure SDK to update storage file share.
        :param old_response: dict with properties of the storage file share
        :return: dict with description of the new storage file share
        """
        self.log('Creating file share {0}'.format(self.name))
        file_share_details = dict(access_tier=self.access_tier if self.access_tier else old_responce.get('access_tier'), share_quota=self.quota if self.quota else old_responce.get('share_quota'), metadata=self.metadata if self.metadata else old_responce.get('metadata'), enabled_protocols=self.enabled_protocols if self.enabled_protocols else old_responce.get('enabled_protocols'), root_squash=self.root_squash if self.root_squash else old_responce.get('self.root_squash'))
        try:
            self.storage_client.file_shares.update(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name, file_share=file_share_details)
        except Exception as e:
            self.fail('Error updating file share {0} : {1}'.format(self.name, str(e)))
        return self.get_share()

    def delete_storage_share(self):
        """
        Method calling the Azure SDK to delete storage share.
        :return: object resulting from the original request
        """
        try:
            self.storage_client.file_shares.delete(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name)
        except Exception as e:
            self.fail('Error deleting file share {0} : {1}'.format(self.name, str(e)))
        return self.get_share()
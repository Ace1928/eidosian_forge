from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMStorageShareInfo(AzureRMModuleBase):
    """
    Info class for an Azure RM Storage share resource
    """

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), account_name=dict(type='str', required=True))
        self.results = dict(changed=False, storageshares=list())
        self.resource_group = None
        self.name = None
        self.account_name = None
        super(AzureRMStorageShareInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        """
        Main module execution method
        """
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name:
            self.results['storageshares'] = self.get_share()
        else:
            self.results['storageshares'] = self.list_all()
        return self.results

    def get_share(self):
        """
        Get the properties of the specified Azure Storage file share.
        :return: dict with properties of the storage file share
        """
        storage_share = None
        try:
            storage_share = self.storage_client.file_shares.get(resource_group_name=self.resource_group, account_name=self.account_name, share_name=self.name)
            self.log('Response : {0}'.format(storage_share))
        except ResourceNotFoundError as e:
            self.log('Did not find the storage share with name {0} : {1}'.format(self.name, str(e)))
        return self.storage_share_to_dict(storage_share)

    def storage_share_to_dict(self, storage_share):
        """
        Transform Azure RM Storage share object to dictionary
        :param storage_share: contains information about storage file share
        :type storage_share: FileShare
        :return: dict generated from storage_share
        """
        return dict(id=storage_share.id, name=storage_share.name, type=storage_share.type, etag=storage_share.etag.replace('"', ''), last_modified_time=storage_share.last_modified_time, metadata=storage_share.metadata, share_quota=storage_share.share_quota, access_tier=storage_share.access_tier, access_tier_change_time=storage_share.access_tier_change_time, enabled_protocols=storage_share.enabled_protocols, root_squash=storage_share.root_squash, version=storage_share.version, deleted=storage_share.deleted, deleted_time=storage_share.deleted_time, remaining_retention_days=storage_share.remaining_retention_days, access_tier_status=storage_share.access_tier_status, share_usage_bytes=storage_share.share_usage_bytes) if storage_share else None

    def list_all(self):
        """
        Method calling the Azure SDK to create storage file share.
        :return: dict with description of the new storage file share
        """
        '\n        Get the properties of the specified Azure Storage file share.\n        :return: dict with properties of the storage file share\n        '
        all_items = None
        try:
            storage_shares = self.storage_client.file_shares.list(resource_group_name=self.resource_group, account_name=self.account_name)
            self.log('Response : {0}'.format(storage_shares))
            all_items = [self.storage_share_to_dict(share) for share in storage_shares]
        except Exception as e:
            self.log('Did not find the storage file share : {0}'.format(str(e)))
        return all_items
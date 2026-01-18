from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.filestore.backups import util as backup_util
from googlecloudsdk.command_lib.filestore.snapshots import util as snapshot_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class FilestoreClient(object):
    """Wrapper for working with the file API."""

    def __init__(self, version=V1_API_VERSION):
        if version == ALPHA_API_VERSION:
            self._adapter = AlphaFilestoreAdapter()
        elif version == BETA_API_VERSION:
            self._adapter = BetaFilestoreAdapter()
        elif version == V1_API_VERSION:
            self._adapter = FilestoreAdapter()
        else:
            raise ValueError('[{}] is not a valid API version.'.format(version))

    @property
    def client(self):
        return self._adapter.client

    @property
    def messages(self):
        return self._adapter.messages

    def ListInstances(self, location_ref, limit=None):
        """Make API calls to List active Cloud Filestore instances.

    Args:
      location_ref: The parsed location of the listed Filestore instances.
      limit: The number of Cloud Filestore instances to limit the results to.
        This limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud Filestore instances.
    """
        request = self.messages.FileProjectsLocationsInstancesListRequest(parent=location_ref)
        response = self.client.projects_locations_instances.List(request)
        for location in response.unreachable:
            log.warning('Location {} may be unreachable.'.format(location))
        return list_pager.YieldFromList(self.client.projects_locations_instances, request, field='instances', limit=limit, batch_size_attribute='pageSize')

    def GetInstance(self, instance_ref):
        """Get Cloud Filestore instance information."""
        request = self.messages.FileProjectsLocationsInstancesGetRequest(name=instance_ref.RelativeName())
        return self.client.projects_locations_instances.Get(request)

    def GetSnapshot(self, snapshot_ref):
        """Get Cloud Filestore snapshot information."""
        request = self.messages.FileProjectsLocationsSnapshotsGetRequest(name=snapshot_ref.RelativeName())
        return self.client.projects_locations_snapshots.Get(request)

    def GetInstanceSnapshot(self, snapshot_ref):
        """Get Cloud Filestore snapshot information."""
        request = self.messages.FileProjectsLocationsInstancesSnapshotsGetRequest(name=snapshot_ref.RelativeName())
        return self.client.projects_locations_instances_snapshots.Get(request)

    def GetBackup(self, backup_ref):
        """Get Cloud Filestore backup information."""
        request = self.messages.FileProjectsLocationsBackupsGetRequest(name=backup_ref.RelativeName())
        return self.client.projects_locations_backups.Get(request)

    def DeleteInstance(self, instance_ref, async_, force=False):
        """Deletes an existing Cloud Filestore instance."""
        request = self.messages.FileProjectsLocationsInstancesDeleteRequest(name=instance_ref.RelativeName(), force=force)
        return self._DeleteInstance(async_, request)

    def DeleteInstanceAlpha(self, instance_ref, async_):
        """Delete an existing Cloud Filestore instance."""
        request = self.messages.FileProjectsLocationsInstancesDeleteRequest(name=instance_ref.RelativeName())
        return self._DeleteInstance(async_, request)

    def _DeleteInstance(self, async_, request):
        delete_op = self.client.projects_locations_instances.Delete(request)
        if async_:
            return delete_op
        operation_ref = resources.REGISTRY.ParseRelativeName(delete_op.name, collection=OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    def GetOperation(self, operation_ref):
        """Gets description of a long-running operation.

    Args:
      operation_ref: the operation reference.

    Returns:
      messages.GoogleLongrunningOperation, the operation.
    """
        request = self.messages.FileProjectsLocationsOperationsGetRequest(name=operation_ref.RelativeName())
        return self.client.projects_locations_operations.Get(request)

    def WaitForOperation(self, operation_ref):
        """Waits on the long-running operation until the done field is True.

    Args:
      operation_ref: the operation reference.

    Raises:
      waiter.OperationError: if the operation contains an error.

    Returns:
      the 'response' field of the Operation.
    """
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), operation_ref, 'Waiting for [{0}] to finish'.format(operation_ref.Name()))

    def CancelOperation(self, operation_ref):
        """Cancels a long-running operation.

    Args:
      operation_ref: the operation reference.

    Returns:
      Empty response message.
    """
        request = self.messages.FileProjectsLocationsOperationsCancelRequest(name=operation_ref.RelativeName())
        return self.client.projects_locations_operations.Cancel(request)

    def CreateInstance(self, instance_ref, async_, config):
        """Create a Cloud Filestore instance."""
        request = self.messages.FileProjectsLocationsInstancesCreateRequest(parent=instance_ref.Parent().RelativeName(), instanceId=instance_ref.Name(), instance=config)
        create_op = self.client.projects_locations_instances.Create(request)
        if async_:
            return create_op
        operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    def GetLocation(self, location_ref):
        request = self.messages.FileProjectsLocationsGetRequest(name=location_ref)
        return self.client.projects_locations.Get(request)

    def ListLocations(self, project_ref, limit=None):
        request = self.messages.FileProjectsLocationsListRequest(name=project_ref.RelativeName())
        return list_pager.YieldFromList(self.client.projects_locations, request, field='locations', limit=limit, batch_size_attribute='pageSize')

    def ListOperations(self, operation_ref, limit=None):
        """Make API calls to List active Cloud Filestore operations.

    Args:
      operation_ref: The parsed location of the listed Filestore instances.
      limit: The number of Cloud Filestore instances to limit the results to.
        This limit is passed to the server and the server does the limiting.

    Returns:
      Generator that yields the Cloud Filestore instances.
    """
        request = self.messages.FileProjectsLocationsOperationsListRequest(name=operation_ref)
        return list_pager.YieldFromList(self.client.projects_locations_operations, request, field='operations', limit=limit, batch_size_attribute='pageSize')

    def ParseFilestoreConfig(self, tier=None, protocol=None, description=None, file_share=None, network=None, labels=None, zone=None, nfs_export_options=None, kms_key_name=None, managed_ad=None):
        """Parses the command line arguments for Create into a config.

    Args:
      tier: The tier.
      protocol: The protocol values are NFS_V3 (default) or NFS_V4_1.
      description: The description of the instance.
      file_share: The config for the file share.
      network: The network for the instance.
      labels: The parsed labels value.
      zone: The parsed zone of the instance.
      nfs_export_options: The nfs export options for the file share.
      kms_key_name: The kms key for instance encryption.
      managed_ad: The Managed Active Directory settings of the instance.

    Returns:
      The configuration that will be used as the request body for creating a
      Cloud Filestore instance.
    """
        instance = self.messages.Instance()
        instance.tier = tier
        if protocol:
            instance.protocol = protocol
        if managed_ad:
            self._adapter.ParseManagedADIntoInstance(instance, managed_ad)
        instance.labels = labels
        if kms_key_name:
            instance.kmsKeyName = kms_key_name
        if description:
            instance.description = description
        if nfs_export_options:
            file_share['nfs_export_options'] = nfs_export_options
        self._adapter.ParseFileShareIntoInstance(instance, file_share, zone)
        if network:
            instance.networks = []
            network_config = self.messages.NetworkConfig()
            network_config.network = network.get('name')
            if 'reserved-ip-range' in network:
                network_config.reservedIpRange = network['reserved-ip-range']
            connect_mode = network.get('connect-mode', 'DIRECT_PEERING')
            self._adapter.ParseConnectMode(network_config, connect_mode)
            instance.networks.append(network_config)
        return instance

    def ParseUpdatedInstanceConfig(self, instance_config, description=None, labels=None, file_share=None, managed_ad=None, disconnect_managed_ad=None, clear_nfs_export_options=False):
        """Parses updates into an instance config.

    Args:
      instance_config: The Instance message to update.
      description: str, a new description, if any.
      labels: LabelsValue message, the new labels value, if any.
      file_share: dict representing a new file share config, if any.
      managed_ad: The Managed Active Directory settings of the instance.
      disconnect_managed_ad: Disconnect from Managed Active Directory.
      clear_nfs_export_options: bool, whether to clear the NFS export options.

    Raises:
      InvalidCapacityError, if an invalid capacity value is provided.
      InvalidNameError, if an invalid file share name is provided.

    Returns:
      The instance message.
    """
        instance = self._adapter.ParseUpdatedInstanceConfig(instance_config, description=description, labels=labels, file_share=file_share, managed_ad=managed_ad, disconnect_managed_ad=disconnect_managed_ad, clear_nfs_export_options=clear_nfs_export_options)
        return instance

    def UpdateInstance(self, instance_ref, instance_config, update_mask, async_):
        """Updates an instance.

    Args:
      instance_ref: the reference to the instance.
      instance_config: Instance message, the updated instance.
      update_mask: str, a comma-separated list of updated fields.
      async_: bool, if False, wait for the operation to complete.

    Returns:
      an Operation or Instance message.
    """
        update_op = self._adapter.UpdateInstance(instance_ref, instance_config, update_mask)
        if async_:
            return update_op
        operation_ref = resources.REGISTRY.ParseRelativeName(update_op.name, collection=OPERATIONS_COLLECTION)
        return self.WaitForOperation(operation_ref)

    @staticmethod
    def MakeNFSExportOptionsMsg(messages, nfs_export_options):
        """Creates an NfsExportOptions message.

    Args:
      messages: The messages module.
      nfs_export_options: list, containing NfsExportOptions dictionary.

    Returns:
      File share message populated with values, filled with defaults.
      In case no nfs export options are provided we rely on the API to apply a
      default.
    """
        read_write = 'READ_WRITE'
        root_squash = 'ROOT_SQUASH'
        no_root_squash = 'NO_ROOT_SQUASH'
        anonimous_uid = 65534
        anonimous_gid = 65534
        nfs_export_configs = []
        if nfs_export_options is None:
            return []
        for nfs_export_option in nfs_export_options:
            access_mode = messages.NfsExportOptions.AccessModeValueValuesEnum.lookup_by_name(nfs_export_option.get('access-mode', read_write))
            squash_mode = messages.NfsExportOptions.SquashModeValueValuesEnum.lookup_by_name(nfs_export_option.get('squash-mode', no_root_squash))
            if nfs_export_option.get('squash-mode', None) == root_squash:
                anon_uid = nfs_export_option.get('anon_uid', anonimous_uid)
                anon_gid = nfs_export_option.get('anon_gid', anonimous_gid)
            else:
                anon_gid = None
                anon_uid = None
            nfs_export_config = messages.NfsExportOptions(ipRanges=nfs_export_option.get('ip-ranges', []), anonUid=anon_uid, anonGid=anon_gid, accessMode=access_mode, squashMode=squash_mode)
            nfs_export_configs.append(nfs_export_config)
        return nfs_export_configs

    @staticmethod
    def MakeNFSExportOptionsMsgBeta(messages, nfs_export_options):
        """Creates an MakeNFSExportOptionsMsgBeta message.

    This function is a copy MakeNFSExportOptionsMsg with addition of handling
    the SecurityFlavors for NFSv41.

    Args:
      messages: The messages module.
      nfs_export_options: list, containing NfsExportOptions dictionary.

    Returns:
      File share message populated with values, filled with defaults.
      In case no nfs export options are provided we rely on the API to apply a
      default.
    """
        read_write = 'READ_WRITE'
        root_squash = 'ROOT_SQUASH'
        no_root_squash = 'NO_ROOT_SQUASH'
        anonimous_uid = 65534
        anonimous_gid = 65534
        nfs_export_configs = []
        if nfs_export_options is None:
            return []
        for nfs_export_option in nfs_export_options:
            access_mode = messages.NfsExportOptions.AccessModeValueValuesEnum.lookup_by_name(nfs_export_option.get('access-mode', read_write))
            squash_mode = messages.NfsExportOptions.SquashModeValueValuesEnum.lookup_by_name(nfs_export_option.get('squash-mode', no_root_squash))
            if nfs_export_option.get('squash-mode', None) == root_squash:
                anon_uid = nfs_export_option.get('anon_uid', anonimous_uid)
                anon_gid = nfs_export_option.get('anon_gid', anonimous_gid)
            else:
                anon_gid = None
                anon_uid = None
            security_flavors_list = []
            flavors = nfs_export_option.get('security-flavors', [])
            for flavor in flavors:
                security_flavors_list.append(messages.NfsExportOptions.SecurityFlavorsValueListEntryValuesEnum.lookup_by_name(flavor))
            nfs_export_config = messages.NfsExportOptions(ipRanges=nfs_export_option.get('ip-ranges', []), anonUid=anon_uid, anonGid=anon_gid, accessMode=access_mode, squashMode=squash_mode, securityFlavors=security_flavors_list)
            nfs_export_configs.append(nfs_export_config)
        return nfs_export_configs
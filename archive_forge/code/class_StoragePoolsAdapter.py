from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class StoragePoolsAdapter(object):
    """Adapter for the Cloud NetApp Files API for Storage Pools."""

    def __init__(self):
        self.release_track = base.ReleaseTrack.GA
        self.client = netapp_api_util.GetClientInstance(release_track=self.release_track)
        self.messages = netapp_api_util.GetMessagesModule(release_track=self.release_track)

    def ParseStoragePoolConfig(self, name, service_level, network, kms_config, active_directory, enable_ldap, capacity, description, allow_auto_tiering, labels):
        """Parses the command line arguments for Create Storage Pool into a config.

    Args:
      name: the name of the Storage Pool
      service_level: the service level of the Storage Pool
      network: the network of the Storage Pool
      kms_config: the KMS Config of the Storage Pool
      active_directory: the Active Directory of the Storage Pool
      enable_ldap: Bool on whether to enable LDAP on Storage Pool
      capacity: the storage capacity of the Storage Pool
      description: the description of the Storage Pool
      allow_auto_tiering: Bool on whether Storage Pool supports auto tiering
      labels: the parsed labels value

    Returns:
      The configuration that will be used as the request body for creating a
      Cloud NetApp Storage Pool.
    """
        storage_pool = self.messages.StoragePool()
        storage_pool.name = name
        storage_pool.serviceLevel = service_level
        storage_pool.network = network.get('name')
        if 'psa-range' in network:
            storage_pool.psaRange = network.get('psa-range')
        storage_pool.kmsConfig = kms_config
        storage_pool.activeDirectory = active_directory
        storage_pool.ldapEnabled = enable_ldap
        storage_pool.capacityGib = capacity
        storage_pool.description = description
        if allow_auto_tiering is not None:
            storage_pool.allowAutoTiering = allow_auto_tiering
        storage_pool.labels = labels
        return storage_pool

    def ParseUpdatedStoragePoolConfig(self, storagepool_config, description=None, active_directory=None, labels=None, capacity=None, allow_auto_tiering=None):
        """Parse update information into an updated Storage Pool message."""
        if capacity is not None:
            storagepool_config.capacityGib = capacity
        if active_directory is not None:
            storagepool_config.activeDirectory = active_directory
        if description is not None:
            storagepool_config.description = description
        if allow_auto_tiering is not None:
            storagepool_config.allowAutoTiering = allow_auto_tiering
        if labels is not None:
            storagepool_config.labels = labels
        return storagepool_config

    def UpdateStoragePool(self, storagepool_ref, storagepool_config, update_mask):
        """Send a Patch request for the Cloud NetApp Storage Pool."""
        update_request = self.messages.NetappProjectsLocationsStoragePoolsPatchRequest(storagePool=storagepool_config, name=storagepool_ref.RelativeName(), updateMask=update_mask)
        update_op = self.client.projects_locations_storagePools.Patch(update_request)
        return update_op
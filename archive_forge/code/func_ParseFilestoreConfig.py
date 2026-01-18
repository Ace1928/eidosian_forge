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
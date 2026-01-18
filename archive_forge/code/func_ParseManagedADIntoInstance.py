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
def ParseManagedADIntoInstance(self, instance, managed_ad):
    """Parses managed-ad configs into an instance message.

    Args:
      instance: The filestore instance struct.
      managed_ad: The managed_ad cli paramters

    Raises:
      InvalidArgumentError: If managed_ad argument constraints are violated.
    """
    domain = managed_ad.get('domain')
    if domain is None:
        raise InvalidArgumentError('Domain parameter is missing in --managed_ad.')
    computer = managed_ad.get('computer')
    if computer is None:
        raise InvalidArgumentError('Computer parameter is missing in --managed_ad.')
    instance.directoryServices = self.messages.DirectoryServicesConfig(managedActiveDirectory=self.messages.ManagedActiveDirectoryConfig(domain=domain, computer=computer))
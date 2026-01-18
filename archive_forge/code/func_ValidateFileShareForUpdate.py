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
def ValidateFileShareForUpdate(self, instance_config, file_share):
    """Validate the updated file share configuration.

    The new config must have the same name as the existing config.

    Args:
      instance_config: Instance message for existing instance.
      file_share: dict with keys 'name' and 'capacity'.

    Raises:
      InvalidNameError: If the names don't match.
      ValueError: If the instance doesn't have an existing file share.
    """
    existing = self.FileSharesFromInstance(instance_config)
    if not existing:
        raise ValueError('Existing instance does not have file shares configured')
    existing_file_share = existing[0]
    if existing_file_share.name != file_share.get('name'):
        raise InvalidNameError('Must update an existing file share. Existing file share is named [{}]. Requested update had name [{}].'.format(existing_file_share.name, file_share.get('name')))
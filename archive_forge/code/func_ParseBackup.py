from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseBackup(self, name=None, source_snapshot=None, source_volume=None, description=None, labels=None):
    """Parses the command line arguments for Create Backup into a message.

    Args:
      name: The name of the Backup.
      source_snapshot: The Source Snapshot of the Backup.
      source_volume: The Source Volume of the Backup.
      description: The description of the Backup.
      labels: The parsed labels value.

    Returns:
      The configuration that will be used ass the request body for creating a
      Cloud NetApp Backup.
    """
    backup = self.messages.Backup()
    backup.name = name
    backup.sourceSnapshot = source_snapshot
    backup.sourceVolume = source_volume
    backup.description = description
    backup.labels = labels
    return backup
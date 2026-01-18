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
def ParseUpdatedBackup(self, backup, description=None, labels=None):
    """Parses updates into a new Backup."""
    if description is not None:
        backup.description = description
    if labels is not None:
        backup.labels = labels
    return backup
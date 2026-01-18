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
def ParseUpdatedReplicationConfig(self, replication_config, description=None, labels=None, replication_schedule=None):
    """Parse update information into an updated Replication message."""
    if description is not None:
        replication_config.description = description
    if labels is not None:
        replication_config.labels = labels
    if replication_schedule is not None:
        replication_config.replicationSchedule = replication_schedule
    return replication_config
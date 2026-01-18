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
def CreateReplication(self, replication_ref, volume_ref, async_, config):
    """Creates a Cloud NetApp Volume Replication."""
    request = self.messages.NetappProjectsLocationsVolumesReplicationsCreateRequest(parent=volume_ref, replicationId=replication_ref.Name(), replication=config)
    create_op = self.client.projects_locations_volumes_replications.Create(request)
    if async_:
        return create_op
    operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)
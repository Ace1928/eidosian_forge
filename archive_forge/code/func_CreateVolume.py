from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateVolume(self, volume_ref, async_, config):
    """Create a Cloud NetApp Volume."""
    request = self.messages.NetappProjectsLocationsVolumesCreateRequest(parent=volume_ref.Parent().RelativeName(), volumeId=volume_ref.Name(), volume=config)
    create_op = self.client.projects_locations_volumes.Create(request)
    if async_:
        return create_op
    operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=constants.OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)
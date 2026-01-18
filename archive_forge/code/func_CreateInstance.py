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
def CreateInstance(self, instance_ref, async_, config):
    """Create a Cloud Filestore instance."""
    request = self.messages.FileProjectsLocationsInstancesCreateRequest(parent=instance_ref.Parent().RelativeName(), instanceId=instance_ref.Name(), instance=config)
    create_op = self.client.projects_locations_instances.Create(request)
    if async_:
        return create_op
    operation_ref = resources.REGISTRY.ParseRelativeName(create_op.name, collection=OPERATIONS_COLLECTION)
    return self.WaitForOperation(operation_ref)
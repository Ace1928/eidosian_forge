from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.compute.ssl_policies import ssl_policies_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.ssl_policies import flags
class DeleteBatchPoller(poller.BatchPoller):

    def __init__(self, compute_adapter, resource_service, target_refs=None):
        super(DeleteBatchPoller, self).__init__(compute_adapter, resource_service, target_refs)

    def GetResult(self, operation_batch):
        return
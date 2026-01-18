from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ast
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def _CreateStopRequest(self, client, instance_ref, args):
    if self.ReleaseTrack() == base.ReleaseTrack.ALPHA:
        return client.messages.ComputeInstancesStopRequest(discardLocalSsd=args.discard_local_ssd, instance=instance_ref.Name(), project=instance_ref.project, zone=instance_ref.zone, noGracefulShutdown=args.no_graceful_shutdown)
    return client.messages.ComputeInstancesStopRequest(discardLocalSsd=args.discard_local_ssd, instance=instance_ref.Name(), project=instance_ref.project, zone=instance_ref.zone)
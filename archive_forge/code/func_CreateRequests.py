from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_services import flags
def CreateRequests(self, args):
    """Creates and returns a BackendServices.DeleteSignedUrlKey request."""
    backend_service_ref = flags.GLOBAL_BACKEND_SERVICE_ARG.ResolveAsResource(args, self.resources, scope_lister=compute_flags.GetDefaultScopeLister(self.compute_client))
    request = self.messages.ComputeBackendServicesDeleteSignedUrlKeyRequest(backendService=backend_service_ref.Name(), keyName=args.key_name, project=self.project)
    return [request]
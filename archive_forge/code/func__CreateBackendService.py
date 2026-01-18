from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import log
def _CreateBackendService(self, holder, args, backend_services_ref):
    health_checks = flags.GetHealthCheckUris(args, self, holder.resources)
    enable_cdn = True if args.enable_cdn else None
    return holder.client.messages.BackendService(description=args.description, name=backend_services_ref.Name(), healthChecks=health_checks, portName=_ResolvePortName(args), protocol=_ResolveProtocol(holder.client.messages, args), timeoutSec=args.timeout, enableCDN=enable_cdn)
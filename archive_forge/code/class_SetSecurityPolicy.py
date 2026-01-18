from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.backend_services import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.command_lib.compute.security_policies import (
@base.Deprecate(is_removed=False, warning='This command is deprecated and will not be promoted to beta. Please use "gcloud beta backend-services update" instead.')
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetSecurityPolicy(base.SilentCommand):
    """Set the security policy for a backend service."""
    SECURITY_POLICY_ARG = None

    @classmethod
    def Args(cls, parser):
        flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.AddArgument(parser)
        cls.SECURITY_POLICY_ARG = security_policy_flags.SecurityPolicyArgumentForTargetResource(resource='backend service', required=True)
        cls.SECURITY_POLICY_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.ResolveAsResource(args, holder.resources)
        if not args.security_policy:
            security_policy_ref = None
        else:
            security_policy_ref = self.SECURITY_POLICY_ARG.ResolveAsResource(args, holder.resources).SelfLink()
        backend_service = client.BackendService(ref, compute_client=holder.client)
        return backend_service.SetSecurityPolicy(security_policy=security_policy_ref)
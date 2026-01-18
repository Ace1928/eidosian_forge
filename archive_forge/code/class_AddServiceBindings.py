from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.backend_services import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AddServiceBindings(base.UpdateCommand):
    """Add service bindings to a backend service."""
    detailed_help = _DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.AddArgument(parser)
        flags.AddServiceBindings(parser, required=True, help_text='List of service binding names to be added to the backend service.')

    def _Modify(self, backend_service_ref, args, existing):
        location = backend_service_ref.region if backend_service_ref.Collection() == 'compute.regionBackendServices' else 'global'
        replacement = encoding.CopyProtoMessage(existing)
        old_bindings = replacement.serviceBindings or []
        new_bindings = [reference_utils.BuildServiceBindingUrl(backend_service_ref.project, location, binding_name) for binding_name in args.service_bindings]
        new_bindings = reference_utils.FilterReferences(new_bindings, old_bindings)
        replacement.serviceBindings = sorted(list(set(old_bindings) | set(new_bindings)))
        return replacement

    def Run(self, args):
        """Adds service bindings to the Backend Service."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        backend_service_ref = flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        backend_service = client.BackendService(backend_service_ref, compute_client=holder.client)
        new_object = self._Modify(backend_service_ref, args, backend_service.Get())
        return backend_service.Set(new_object)
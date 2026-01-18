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
class RemoveServiceBindings(base.UpdateCommand):
    """Remove service bindings from a backend service."""
    detailed_help = _DETAILED_HELP

    @classmethod
    def Args(cls, parser):
        flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.AddArgument(parser)
        flags.AddServiceBindings(parser, required=True, help_text='List of service binding names to be removed from the backend service.')

    def _Modify(self, backend_service_ref, args, existing):
        location = backend_service_ref.region if backend_service_ref.Collection() == 'compute.regionBackendServices' else 'global'
        replacement = encoding.CopyProtoMessage(existing)
        old_bindings = replacement.serviceBindings or []
        bindings_to_remove = [reference_utils.BuildServiceBindingUrl(backend_service_ref.project, location, binding_name) for binding_name in args.service_bindings]
        replacement.serviceBindings = reference_utils.FilterReferences(old_bindings, bindings_to_remove)
        return replacement

    def Run(self, args):
        """Remove service bindings from the Backend Service."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        backend_service_ref = flags.GLOBAL_REGIONAL_BACKEND_SERVICE_ARG.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        backend_service = client.BackendService(backend_service_ref, compute_client=holder.client)
        new_object = self._Modify(backend_service_ref, args, backend_service.Get())
        cleared_fields = []
        if not new_object.serviceBindings:
            cleared_fields.append('serviceBindings')
        with holder.client.apitools_client.IncludeFields(cleared_fields):
            return backend_service.Set(new_object)
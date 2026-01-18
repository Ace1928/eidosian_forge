from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class InstancesPatchPartnerMetadata(base.UpdateCommand):
    """patch partner metadata."""

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='set partner metadata on')
        partner_metadata_utils.AddPartnerMetadataArgs(parser)

    def _make_patch_partner_metadata_request(self, client, instance_ref, args):
        partner_metadata_dict = partner_metadata_utils.CreatePartnerMetadataDict(args)
        partner_metadata_message = partner_metadata_utils.ConvertPartnerMetadataDictToMessage(partner_metadata_dict)
        return (client.apitools_client.instances, 'PatchPartnerMetadata', client.messages.ComputeInstancesPatchPartnerMetadataRequest(partnerMetadata=client.messages.PartnerMetadata(partnerMetadata=partner_metadata_message), **instance_ref.AsDict()))

    def Run(self, args):
        if not args.partner_metadata and (not args.partner_metadata_from_file):
            raise calliope_exceptions.OneOfArgumentsRequiredException(['--partner-metadata', '--partner-metadata-from-file'], 'At least one of [--partner-metadata] or [--partner-metadata-from-file] must be provided.')
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        patch_request = self._make_patch_partner_metadata_request(client, instance_ref, args)
        return client.MakeRequests([patch_request])
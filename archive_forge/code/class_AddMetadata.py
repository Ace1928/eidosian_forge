from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_settings_metadata_utils as metadata_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.project_zonal_metadata import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class AddMetadata(base.UpdateCommand):
    """Add or update project zonal metadata.

  *{command}* is used to add or update project zonal metadata for your VM
  instances. Project zonal metadata values propagate to all VMs within the
  specified zone. Every VM has access to a metadata server that you can use to
  query the configured project zonal metadata values. To set metadata for
  individual instances, use `gcloud compute instances add-metadata`. For
  information about metadata, see
  [](https://cloud.google.com/compute/docs/metadata).

  Only the metadata keys that you provide in the command get mutated. All other
  existing metadata entries remain the same.
  """
    detailed_help = {'EXAMPLES': "        To set the project zonal metadata with key=value in the zone ``us-central1-a''\n        for the project ``my-gcp-project'', run:\n\n        $ {command} --metadata=key=value\n        --zone=us-central1-a --project=my-gcp-project\n\n        For more information and examples for setting project zonal metadata, see\n        [](https://cloud.google.com/compute/docs/metadata/setting-custom-metadata#set-custom-project-zonal-metadata)\n      "}

    @staticmethod
    def Args(parser):
        flags.ProjectZonalMetadataAddMetadataFlags(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        service = client.apitools_client.instanceSettings
        get_request = client.messages.ComputeInstanceSettingsGetRequest(project=properties.VALUES.core.project.GetOrFail(), zone=args.zone)
        existing_instance_settings = client.MakeRequests([(service, 'Get', get_request)])[0]
        fingerprint = existing_instance_settings.fingerprint
        if metadata_utils.IsRequestMetadataSameAsExistingMetadata(args.metadata, metadata_utils.ConstructMetadataDict(existing_instance_settings.metadata)):
            log.status.Print('No change requested; skipping update for project zonal metadata in zone [{0}].'.format(existing_instance_settings.zone))
            return existing_instance_settings.metadata
        request = client.messages.ComputeInstanceSettingsPatchRequest(instanceSettings=client.messages.InstanceSettings(fingerprint=fingerprint, metadata=metadata_utils.ConstructInstanceSettingsMetadataMessage(client.messages, args.metadata)), project=properties.VALUES.core.project.GetOrFail(), updateMask=metadata_utils.ConstructUpdateMask(args.metadata.keys()), zone=args.zone)
        return client.MakeRequests([(service, 'Patch', request)], no_followup=True)[0]
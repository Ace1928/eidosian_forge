from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_settings import flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class SetServiceAccount(base.UpdateCommand):
    """Set service account in zonal instance settings."""
    detailed_help = {'EXAMPLES': "\n        To update the instance settings in the zone called ``us-central1-a''\n        in the project ``my-gcp-project'' with service account email ``example@serviceaccount.com'', run:\n\n          $ {command} example@serviceaccount.com --zone=us-central1-a --project=my-gcp-project\n      "}

    @staticmethod
    def Args(parser):
        flags.AddServiceAccountFlags(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        service = client.apitools_client.instanceSettings
        get_request = client.messages.ComputeInstanceSettingsGetRequest(project=properties.VALUES.core.project.GetOrFail(), zone=args.zone)
        fingerprint = client.MakeRequests([(service, 'Get', get_request)])[0].fingerprint
        request = client.messages.ComputeInstanceSettingsPatchRequest(instanceSettings=client.messages.InstanceSettings(email=getattr(args, 'SERVICE_ACCOUNT_EMAIL', None), fingerprint=fingerprint), project=properties.VALUES.core.project.GetOrFail(), updateMask='email', zone=args.zone)
        return client.MakeRequests([(service, 'Patch', request)], no_followup=True)[0]
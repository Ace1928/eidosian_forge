from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class OsUpgrade(base.Command):
    """Upgrade a instance's OS version."""

    @classmethod
    def Args(cls, parser):
        parser.add_argument('--source-os', required=True, choices=sorted(_OS_CHOICES_SOURCE), help='OS version of the source instance to upgrade.')
        parser.add_argument('--target-os', required=True, choices=sorted(_OS_CHOICES_TARGET), help='Version of the OS after upgrade.')
        parser.add_argument('--create-machine-backup', required=False, action='store_true', default=True, help='When enabled, a machine image is created that backs up the original state of your instance.')
        parser.add_argument('--auto-rollback', required=False, action='store_true', help='When auto rollback is enabled, the instance and its resources are restored to their original state. Otherwise, the instance and any temporary resources are left in the intermediate state of the time of failure. This is useful for debugging.')
        parser.add_argument('--use-staging-install-media', required=False, action='store_true', help='Use staging install media. This flag is for testing only. Set to true to upgrade with staging windows install media.', hidden=True)
        daisy_utils.AddCommonDaisyArgs(parser, operation='an upgrade')
        daisy_utils.AddExtraCommonDaisyArgs(parser)
        flags.INSTANCES_ARG_FOR_OS_UPGRADE.AddArgument(parser, operation_type=_OS_UPGRADE_OPERATION_TYPE)

    def _ValidateArgs(self, args, compute_client):
        daisy_utils.ValidateZone(args, compute_client)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        self._ValidateArgs(args, client)
        ref = flags.INSTANCES_ARG_FOR_OS_UPGRADE.ResolveAsResource(args, holder.resources, scope_lister=instances_flags.GetInstanceZoneScopeLister(client))
        instance_uri = 'projects/{0}/zones/{1}/instances/{2}'.format(ref.project, ref.zone, ref.Name())
        _PromptForUpgrade(ref, args)
        args.zone = ref.zone
        log.warning('Upgrading OS. This usually takes around 40 minutes but may take up to 90 minutes.')
        return daisy_utils.RunOsUpgradeBuild(args=args, output_filter=_OUTPUT_FILTER, instance_uri=instance_uri, release_track=self.ReleaseTrack().id.lower() if self.ReleaseTrack() else None)
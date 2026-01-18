from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.rollouts import flags as rollout_flags
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as alpha_messages
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Pause(base.UpdateCommand):
    """Pause a rollout resource."""
    detailed_help = {'EXAMPLES': _EXAMPLES}

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        """Registers flags for the pause command."""
        flags = rollout_flags.RolloutFlags(parser)
        flags.AddRolloutResourceArg()
        flags.AddAsync()

    def Run(self, args: parser_extensions.Namespace) -> alpha_messages.Operation:
        """Runs the pause command."""
        flag_parser = rollout_flags.RolloutFlagParser(args, release_track=base.ReleaseTrack.ALPHA)
        req = alpha_messages.GkehubProjectsLocationsRolloutsPauseRequest()
        req.name = util.RolloutName(args)
        req.pauseRolloutRequest = alpha_messages.PauseRolloutRequest()
        fleet_client = client.FleetClient(release_track=self.ReleaseTrack())
        operation = fleet_client.PauseRollout(req)
        rollout_ref = util.RolloutRef(args)
        if flag_parser.Async():
            log.Print('Pause in progress for Fleet rollout [{}]'.format(rollout_ref.SelfLink()))
            return operation
        operation_client = client.OperationClient(release_track=base.ReleaseTrack.ALPHA)
        completed_operation = operation_client.Wait(util.OperationRef(operation))
        log.Print('Paused Fleet rollout [{}].'.format(rollout_ref.SelfLink()))
        return completed_operation
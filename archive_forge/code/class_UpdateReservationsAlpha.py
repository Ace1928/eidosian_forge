from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.commitments import flags
from googlecloudsdk.command_lib.compute.commitments import reservation_helper
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UpdateReservationsAlpha(base.UpdateCommand):
    """Update the resource shape of reservations within the commitment."""
    detailed_help = {'EXAMPLES': "\n        To update reservations of the commitment called ``commitment-1'' in\n        the ``us-central1'' region with values from ``reservations.yaml'', run:\n\n          $ {command} commitment-1 --reservations-from-file=reservations.yaml\n\n        For detailed examples, please refer to [](https://cloud.google.com/compute/docs/instances/reserving-zonal-resources#modifying_reservations_that_are_attached_to_commitments)\n      "}

    @staticmethod
    def Args(parser):
        flags.MakeCommitmentArg(False).AddArgument(parser, operation_type='update reservation')
        flags.AddUpdateReservationGroup(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resources = holder.resources
        commitment_ref = flags.MakeCommitmentArg(False).ResolveAsResource(args, resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        service = client.apitools_client.regionCommitments
        messages = client.messages
        update_reservation_request = messages.RegionCommitmentsUpdateReservationsRequest(reservations=reservation_helper.MakeUpdateReservations(args, messages, resources))
        request = messages.ComputeRegionCommitmentsUpdateReservationsRequest(commitment=commitment_ref.Name(), project=commitment_ref.project, region=commitment_ref.region, regionCommitmentsUpdateReservationsRequest=update_reservation_request)
        return client.MakeRequests([(service, 'UpdateReservations', request)])
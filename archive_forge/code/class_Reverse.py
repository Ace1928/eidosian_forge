from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp.volumes.replications import client as replications_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp.volumes.replications import flags as replications_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Reverse(base.Command):
    """Reverse a Cloud NetApp Volume Replication's direction."""
    _RELEASE_TRACK = base.ReleaseTrack.GA
    detailed_help = {'DESCRIPTION': '          Reverse a Cloud NetApp Volume Replication.\n          ', 'EXAMPLES': '          The following command reverses a Replication named NAME using the required arguments:\n\n              $ {command} NAME --location=us-central1 --volume=vol1\n\n          To reverse a Replication named NAME asynchronously, run the following command:\n\n              $ {command} NAME --location=us-central1 --volume=vol1 --async\n          '}

    @staticmethod
    def Args(parser):
        concept_parsers.ConceptParser([flags.GetReplicationPresentationSpec('The Replication to reverse direction.')]).AddToParser(parser)
        replications_flags.AddReplicationVolumeArg(parser, reverse_op=True)
        flags.AddResourceAsyncFlag(parser)

    def Run(self, args):
        """Reverse a Cloud NetApp Volume Replication's direction in the current project."""
        replication_ref = args.CONCEPTS.replication.Parse()
        client = replications_client.ReplicationsClient(self._RELEASE_TRACK)
        result = client.ReverseReplication(replication_ref, args.async_)
        if args.async_:
            command = 'gcloud {} netapp volumes replications list'.format(self.ReleaseTrack().prefix)
            log.status.Print('Check the status of the reversed replication by listing all replications:\n  $ {} '.format(command))
        return result
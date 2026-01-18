from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
@base.Hidden
class ChangeQuorum(base.Command):
    """Change quorum of a Cloud Spanner database."""
    detailed_help = {'EXAMPLES': textwrap.dedent('        To trigger change quorum from single-region mode to dual-region mode, run:\n\n          $ {command} my-database-id --instance=my-instance-id --dual-region\n\n        To trigger change quorum from dual-region mode to single-region mode with serving location as `asia-south1`, run:\n\n          $ {command} my-database-id --instance=my-instance-id --single-region --serving-location=asia-south1\n\n        To trigger change quorum using etag specified, run:\n\n          $ {command} my-database-id --instance=my-instance-id --dual-region --etag=ETAG\n        ')}

    @staticmethod
    def Args(parser):
        """See base class."""
        resource_args.AddDatabaseResourceArg(parser, 'to change quorum')
        dual_region_or_single_region = parser.add_mutually_exclusive_group(required=True)
        dual_region_flags = dual_region_or_single_region.add_argument_group('Command-line flag for dual-region quorum change:')
        dual_region_flags.add_argument('--dual-region', required=True, action='store_true', help='Switch to dual-region quorum type.')
        single_region_flags = dual_region_or_single_region.add_argument_group('Command-line flags for single-region quorum change:')
        single_region_flags.add_argument('--single-region', required=True, action='store_true', help='Switch to single-region quorum type.')
        single_region_flags.add_argument('--serving-location', required=True, help='The cloud Spanner location.')
        parser.add_argument('--etag', help='Used for optimistic concurrency control.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        msgs = apis.GetMessagesModule('spanner', 'v1')
        if args.dual_region:
            quorum_type = msgs.QuorumType(dualRegion=msgs.DualRegionQuorum())
        else:
            quorum_type = msgs.QuorumType(singleRegion=msgs.SingleRegionQuorum(servingLocation=args.serving_location))
        return databases.ChangeQuorum(args.CONCEPTS.database.Parse(), quorum_type, args.etag)
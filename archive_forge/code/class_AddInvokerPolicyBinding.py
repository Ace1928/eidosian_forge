from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import util
from googlecloudsdk.command_lib.functions.v2.add_invoker_policy_binding import command as command_v2
from googlecloudsdk.command_lib.iam import iam_util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AddInvokerPolicyBinding(util.FunctionResourceCommand):
    """Adds an invoker binding to the IAM policy of a Google Cloud Function.

  This command applies to Cloud Functions 2nd gen only.
  """
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Registers flags for this command."""
        flags.AddFunctionResourceArg(parser, 'to add the invoker binding to')
        flags.AddGen2Flag(parser, 'to add the invoker binding to', hidden=True)
        iam_util.AddMemberFlag(parser, 'to add to the IAM policy', False)

    def _RunV1(self, args: parser_extensions.Namespace):
        return api_util_v1.AddFunctionIamPolicyBinding(args.CONCEPTS.name.Parse().RelativeName(), member=args.member, role='roles/cloudfunctions.invoker')

    def _RunV2(self, args):
        return command_v2.Run(args, self.ReleaseTrack())
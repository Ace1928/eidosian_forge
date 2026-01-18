from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.billing import billing_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.billing import flags
from googlecloudsdk.command_lib.billing import utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class LinkAlpha(base.Command):
    """Link a project with a billing account."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        account_args_group = parser.add_mutually_exclusive_group(required=True)
        flags.GetOldAccountIdArgument(positional=False).AddToParser(account_args_group)
        flags.GetAccountIdArgument(positional=False).AddToParser(account_args_group)
        flags.GetProjectIdArgument().AddToParser(parser)

    def Run(self, args):
        return _RunLink(args)
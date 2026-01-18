from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class EnableAlpha(base.SilentCommand):
    """Enables a service for consumption for a project, folder or organization."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        common_flags.available_service_flag(suffix='to enable').AddToParser(parser)
        common_flags.add_resource_args(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        common_flags.validate_only_args(parser)

    def Run(self, args):
        """Run 'services enable'.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Nothing.
    """
        if args.IsSpecified('project'):
            project = args.project
        else:
            project = properties.VALUES.core.project.Get(required=True)
        if args.IsSpecified('folder'):
            folder = args.folder
        else:
            folder = None
        if args.IsSpecified('organization'):
            organization = args.organization
        else:
            organization = None
        op = serviceusage.AddEnableRule(args.service, project, folder=folder, organization=organization, validate_only=args.validate_only)
        if not args.validate_only:
            if args.async_:
                cmd = _OP_WAIT_CMD.format(op.name)
                log.status.Print('Asynchronous operation is in progress... Use the following command to wait for its completion:\n {0}'.format(cmd))
        log.status.Print('Operation finished successfully')
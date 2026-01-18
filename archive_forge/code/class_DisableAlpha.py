from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import arg_parsers
from googlecloudsdk.command_lib.services import common_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DisableAlpha(base.SilentCommand):
    """Disable a service for consumption for a project, folder or organization.

  This command disables one or more previously-enabled services for
  consumption.

  To see a list of the enabled services for a project, run:

    $ {parent_command} list

  More information on listing services can be found at:
  https://cloud.google.com/service-usage/docs/list-services and on
  disabling a service at:
  https://cloud.google.com/service-usage/docs/enable-disable

  ## EXAMPLES
  To disable a service called `my-consumed-service` for the current
  project, run:

    $ {command} my-consumed-service

  To disable a service called `my-consumed-service` for the project
  `my-project`, run:

    $ {command} my-consumed-service --project=my-project

  To disable a service called `my-consumed-service` for the folder
  `my-folder`, run:

    $ {command} my-consumed-service --folder=my-folder

  To disable a service called `my-consumed-service` for the organization
  `my-organization`, run:

    $ {command} my-consumed-service --organization=my-organization

  To run the same command asynchronously (non-blocking), run:

    $ {command} my-consumed-service --async
  """

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        common_flags.consumer_service_flag(suffix='to disable').AddToParser(parser)
        common_flags.add_resource_args(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        common_flags.validate_only_args(parser)
        parser.add_argument('--force', action='store_true', help='If specified, the disable call will proceed even if there are enabled services which depend on the service to be disabled. Forcing the call means that the services which depend on the service to be disabled will also be disabled.')

    def Run(self, args):
        """Run 'services disable'.

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
        for service_name in args.service:
            service_name = arg_parsers.GetServiceNameFromArg(service_name)
            protected_msg = serviceusage.GetProtectedServiceWarning(service_name)
            if protected_msg:
                if args.IsSpecified('quiet'):
                    raise console_io.RequiredPromptError()
                do_disable = console_io.PromptContinue(protected_msg, default=False, throw_if_unattended=True)
                if not do_disable:
                    continue
            op = serviceusage.RemoveEnableRule(project, service_name, force=args.force, folder=folder, organization=organization, validate_only=args.validate_only)
            if not args.validate_only:
                if op.done:
                    continue
                if args.async_:
                    cmd = OP_WAIT_CMD.format(op.name)
                    log.status.Print('Asynchronous operation is in progress... Use the following command to wait for its completion:\n {0}'.format(cmd))
                    continue
        log.status.Print('Operation finished successfully')
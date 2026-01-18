from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import execution_utils
class Workload(base.Command):
    """Generate gRPC traffic for a given sample app's backend service.

  Before sending traffic to the backend service, create the database and
  start the service with:

      $ {parent_command} init APPNAME --instance-id=INSTANCE_ID
      $ {parent_command} backend APPNAME --instance-id=INSTANCE_ID

  To run all three steps together, use:

      $ {parent_command} run APPNAME --instance-id=INSTANCE_ID
  """
    detailed_help = {'EXAMPLES': textwrap.dedent("          To generate traffic for the 'finance' sample app, run:\n\n          $ {command} finance\n        ")}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        parser.add_argument('appname', help='The sample app name, e.g. "finance".')
        parser.add_argument('--duration', default='1h', type=arg_parsers.Duration(), help='Duration of time allowed to run before stopping the workload.')
        parser.add_argument('--port', type=int, help='Port of the running backend service.')
        parser.add_argument('--target-qps', type=int, help='Target requests per second.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        proc = run_workload(args.appname, args.port)
        try:
            with execution_utils.RaisesKeyboardInterrupt():
                return proc.wait(args.duration)
        except KeyboardInterrupt:
            proc.terminate()
            return 'Workload generator killed'
        except execution_utils.TIMEOUT_EXPIRED_ERR:
            proc.terminate()
            return 'Workload generator killed after {duration}s'.format(duration=args.duration)
        return
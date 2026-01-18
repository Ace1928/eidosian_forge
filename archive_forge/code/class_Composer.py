from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Composer(base.Group):
    """Create and manage Cloud Composer Environments.

  Cloud Composer is a managed Apache Airflow service that helps you create,
  schedule, monitor and manage workflows. Cloud Composer automation helps you
  create Airflow environments quickly and use Airflow-native tools, such as the
  powerful Airflow web interface and command line tools, so you can focus on
  your workflows and not your infrastructure.

  ## EXAMPLES

  To see how to create and manage environments, run:

      $ {command} environments --help

  To see how to manage long-running operations, run:

      $ {command} operations --help
  """
    category = base.DATA_ANALYTICS_CATEGORY

    def Filter(self, context, args):
        """Modify the context that will be given to this group's commands when run.

    Args:
      context: {str:object}, A set of key-value pairs that can be used for
          common initialization among commands.
      args: argparse.Namespace: The same namespace given to the corresponding
          .Run() invocation.

    Returns:
      The refined command context.
    """
        base.RequireProjectID(args)
        base.DisableUserProjectQuota()
        return context
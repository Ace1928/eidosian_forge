from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def HttpHealthCheckArgumentForTargetPool(action, required=True):
    return compute_flags.ResourceArgument(resource_name='http health check', name='--http-health-check', completer=compute_completers.HttpHealthChecksCompleter, plural=False, required=required, global_collection='compute.httpHealthChecks', short_help='Specifies an HTTP health check object to {0} the target pool.'.format(action))
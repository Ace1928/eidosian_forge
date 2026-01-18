from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def HttpHealthCheckArgumentForTargetPoolCreate(required=True):
    return compute_flags.ResourceArgument(resource_name='http health check', name='--http-health-check', completer=compute_completers.HttpHealthChecksCompleter, plural=False, required=required, global_collection='compute.httpHealthChecks', short_help='Specifies HttpHealthCheck to determine the health of instances in the pool.', detailed_help='        Specifies an HTTP health check resource to use to determine the health\n        of instances in this pool. If no health check is specified, traffic will\n        be sent to all instances in this target pool as if the instances\n        were healthy, but the health status of this pool will appear as\n        unhealthy as a warning that this target pool does not have a health\n        check.\n        ')
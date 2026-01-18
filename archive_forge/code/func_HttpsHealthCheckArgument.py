from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def HttpsHealthCheckArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='HTTPS health check', completer=compute_completers.HttpsHealthChecksCompleter, plural=plural, required=required, global_collection='compute.httpsHealthChecks')
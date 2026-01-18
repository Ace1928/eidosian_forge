from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import health_checks_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.health_checks import flags
def _DetailedHelp():
    return {'brief': 'Create a HTTPS health check to monitor load balanced instances', 'DESCRIPTION': '        *{command}* is used to create a non-legacy health check using the HTTPS\n        protocol. You can use this health check for Google Cloud\n        load balancers or for managed instance group autohealing. For more\n        information, see the health checks overview at:\n        [](https://cloud.google.com/load-balancing/docs/health-check-concepts)\n        '}
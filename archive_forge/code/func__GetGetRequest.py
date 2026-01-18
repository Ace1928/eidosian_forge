from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import health_checks_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.health_checks import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
def _GetGetRequest(client, health_check_ref):
    """Returns a request for fetching the existing health check."""
    return (client.apitools_client.healthChecks, 'Get', client.messages.ComputeHealthChecksGetRequest(healthCheck=health_check_ref.Name(), project=health_check_ref.project))
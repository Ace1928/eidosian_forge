from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.http_health_checks import flags
from googlecloudsdk.core import log
def GetSetRequest(self, client, http_health_check_ref, replacement):
    """Returns a request for updated the HTTP health check."""
    return (client.apitools_client.httpHealthChecks, 'Update', client.messages.ComputeHttpHealthChecksUpdateRequest(httpHealthCheck=http_health_check_ref.Name(), httpHealthCheckResource=replacement, project=http_health_check_ref.project))
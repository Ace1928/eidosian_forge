from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def CreateLogConfig(client, args):
    """Returns a HealthCheckLogconfig message if args are valid."""
    messages = client.messages
    log_config = None
    if args.enable_logging is not None:
        log_config = messages.HealthCheckLogConfig(enable=args.enable_logging)
    return log_config
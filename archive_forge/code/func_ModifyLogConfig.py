from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def ModifyLogConfig(client, args, existing_log_config):
    """Returns a modified HealthCheckLogconfig message."""
    messages = client.messages
    log_config = None
    if not existing_log_config:
        if args.enable_logging is None:
            return log_config
        log_config = messages.HealthCheckLogConfig()
    else:
        log_config = copy.deepcopy(existing_log_config)
    if args.enable_logging is not None:
        log_config.enable = args.enable_logging
    return log_config
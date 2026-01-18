from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddHealthCheckLoggingRelatedArgs(parser):
    """Adds parser arguments for health check log config."""
    parser.add_argument('--enable-logging', action='store_true', default=None, help='      Enable logging of health check probe results to Stackdriver. Logging is\n      disabled by default.\n\n      Use --no-enable-logging to disable logging.')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddHealthCheckSourceRegionsRelatedArgs(parser):
    """Adds parser arguments for health check source regions."""
    parser.add_argument('--source-regions', metavar='REGION', help='        Define the list of Google Cloud regions from which health checks are\n        performed. This option is supported only for global health checks that\n        will be referenced by DNS routing policies. If specified, the\n        --check-interval field should be at least 30 seconds. The\n        --proxy-header and --request fields (for TCP health checks) are not\n        supported with this option.\n\n        If --source-regions is specified for a health check, then that health\n        check cannot be used by a backend service or by a managed instance\n        group (for autohealing).\n        ', type=arg_parsers.ArgList(min_length=3), default=[])
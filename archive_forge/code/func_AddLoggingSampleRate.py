from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddLoggingSampleRate(parser):
    """Adds the logging sample rate argument to the argparse."""
    parser.add_argument('--logging-sample-rate', type=arg_parsers.BoundedFloat(lower_bound=0.0, upper_bound=1.0), help='      This field can only be specified if logging is enabled for the backend\n      service. The value of the field must be a float in the range [0, 1]. This\n      configures the sampling rate of requests to the load balancer where 1.0\n      means all logged requests are reported and 0.0 means no logged requests\n      are reported. The default value is 1.0 when logging is enabled and 0.0\n      otherwise.\n      ')
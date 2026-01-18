from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddTimeoutsArgs(parser, for_create=False):
    """Adds arguments to specify connection timeouts."""
    _AddClearableArgument(parser, for_create, 'udp-idle-timeout', arg_parsers.Duration(), textwrap.dedent('         Timeout for UDP connections. See\n         https://cloud.google.com/sdk/gcloud/reference/topic/datetimes for\n         information on duration formats.'), 'Clear timeout for UDP connections')
    _AddClearableArgument(parser, for_create, 'icmp-idle-timeout', arg_parsers.Duration(), textwrap.dedent('         Timeout for ICMP connections. See\n         https://cloud.google.com/sdk/gcloud/reference/topic/datetimes for\n         information on duration formats.'), 'Clear timeout for ICMP connections')
    _AddClearableArgument(parser, for_create, 'tcp-established-idle-timeout', arg_parsers.Duration(), textwrap.dedent('         Timeout for TCP established connections. See\n         https://cloud.google.com/sdk/gcloud/reference/topic/datetimes for\n         information on duration formats.'), 'Clear timeout for TCP established connections')
    _AddClearableArgument(parser, for_create, 'tcp-transitory-idle-timeout', arg_parsers.Duration(), textwrap.dedent('         Timeout for TCP transitory connections. See\n         https://cloud.google.com/sdk/gcloud/reference/topic/datetimes for\n         information on duration formats.'), 'Clear timeout for TCP transitory connections')
    _AddClearableArgument(parser, for_create, 'tcp-time-wait-timeout', arg_parsers.Duration(), textwrap.dedent('         Timeout for TCP connections in the TIME_WAIT state. See\n         https://cloud.google.com/sdk/gcloud/reference/topic/datetimes for\n         information on duration formats.'), 'Clear timeout for TCP connections in the TIME_WAIT state')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def _GetPreference():
    """Returns the --preference flag value choices name:description dict."""
    preferences = {'DEFAULT': textwrap.dedent("\n          This is the default setting. If the designated preferred backends\n          don't have enough capacity, backends in the default category are used.\n          Traffic is distributed between default backends based on the load\n          balancing algorithm you use.\n          "), 'PREFERRED': textwrap.dedent('\n          Backends with this preference setting are used up to their capacity\n          limits first, while optimizing overall network latency.\n          ')}
    return preferences
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def CreateDeprecationAction(name):
    return actions.DeprecationAction(name, warn="The {flag_name} option is deprecated and will be removed in an upcoming release. If you're currently using this argument, you should remove it from your workflows.", removed=False, action='store')
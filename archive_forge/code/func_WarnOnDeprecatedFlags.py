from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def WarnOnDeprecatedFlags(args):
    if getattr(args, 'zone', None):
        log.warning('The --zone flag is deprecated, please use --instance-group-zone instead. It will be removed in a future release.')
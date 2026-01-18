from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def _JoinTypes(types):
    return ', or '.join([', '.join(types[:-1]), types[-1]]) if len(types) > 2 else ' or '.join(types)
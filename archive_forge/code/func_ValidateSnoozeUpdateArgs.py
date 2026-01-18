from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def ValidateSnoozeUpdateArgs(args):
    """Validate snooze update args."""
    if args.fields and (not args.snooze_from_file):
        raise exceptions.OneOfArgumentsRequiredException(['--snooze-from-file'], 'If --fields is specified.')
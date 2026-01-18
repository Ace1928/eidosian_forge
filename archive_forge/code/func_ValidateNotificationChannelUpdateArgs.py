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
def ValidateNotificationChannelUpdateArgs(args):
    """Validate notification channel update args."""
    if args.fields and (not (args.channel_content or args.channel_content_from_file)):
        raise exceptions.OneOfArgumentsRequiredException(['--channel-content', '--channel-content-from-file'], 'If --fields is specified.')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.util import completers
def ValidateSourceArgs(args, sources):
    """Validate that there is one, and only one, source for creating an image."""
    sources_error_message = 'Please specify a source for image creation.'
    source_arg_list = [getattr(args, s.replace('-', '_')) for s in sources]
    source_arg_count = sum((bool(a) for a in source_arg_list))
    source_arg_names = ['--' + s for s in sources]
    if source_arg_count > 1:
        raise exceptions.ConflictingArgumentsException(*source_arg_names)
    if source_arg_count < 1:
        raise exceptions.MinimumArgumentException(source_arg_names, sources_error_message)
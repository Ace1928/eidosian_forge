from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def AddUpdateLabelsFlags(parser, extra_update_message='', extra_remove_message='', enable_clear=True):
    """Adds update command labels flags to an argparse parser.

  Args:
    parser: The argparse parser to add the flags to.
    extra_update_message: str, extra message to append to help text for
                          --update-labels flag.
    extra_remove_message: str, extra message to append to help text for
                          --delete-labels flag.
    enable_clear: bool, whether to include the --clear-labels flag.
  """
    GetUpdateLabelsFlag(extra_update_message).AddToParser(parser)
    if enable_clear:
        remove_group = parser.add_mutually_exclusive_group()
        GetClearLabelsFlag().AddToParser(remove_group)
        GetRemoveLabelsFlag(extra_remove_message).AddToParser(remove_group)
    else:
        GetRemoveLabelsFlag(extra_remove_message).AddToParser(parser)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetLinkType(messages, link_type_arg):
    """Converts the link type flag to a message enum.

  Args:
    messages: The API messages holder.
    link_type_arg: The link type flag value.
  Returns:
    An LinkTypeValueValuesEnum of the flag value, or None if absent.
  """
    if link_type_arg is None:
        return None
    else:
        return messages.Interconnect.LinkTypeValueValuesEnum(link_type_arg)
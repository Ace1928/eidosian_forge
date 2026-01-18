from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
Parses a string action into a FirewallAction and returns it.

  Actions are parsed to be one of AllowAction, BlockAction, RedirectAction,
  SubstituteAction or SetHeaderAction.

  Args:
    action: The action string to parse.
    messages: The message module in which FirewallAction is found in the cloud
      API.

  Returns:
    An instance of FirewallAction containing the action represented in the given
    string.

  Raises:
    BadArgumentException: A parsing error occurred.
  
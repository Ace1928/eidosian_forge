from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
def ExecuteCommandDoNotUse(self, args):
    """Execute a command using the given CLI.

    Do not introduce new invocations of this method unless your command
    *requires* it; any such new invocations must be approved by a team lead.

    Args:
      args: list of str, the args to Execute() via the CLI.

    Returns:
      pass-through of the return value from Execute()
    """
    return self._cli_power_users_only.Execute(args, call_arg_complete=False)
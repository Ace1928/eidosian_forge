from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse
import collections
import io
import itertools
import os
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base  # pylint: disable=unused-import
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import suggest_commands
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
import six
def MakeGetOrRaise(self, flag_name):
    """Returns a function to get given flag value or raise if it is not set.

    This is useful when given flag becomes required when another flag
    is present.

    Args:
      flag_name: str, The flag_name name for the arg to check.

    Raises:
      parser_errors.RequiredError: if flag is not specified.
      UnknownDestinationException: If there is no registered arg for flag_name.

    Returns:
      Function for accessing given flag value.
    """

    def _Func():
        flag = flag_name[2:] if flag_name.startswith('--') else flag_name
        flag_value = getattr(self, flag)
        if flag_value is None and (not self.IsSpecified(flag)):
            raise parser_errors.RequiredError(argument=flag_name)
        return flag_value
    return _Func
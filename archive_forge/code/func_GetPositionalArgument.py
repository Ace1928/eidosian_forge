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
def GetPositionalArgument(self, name):
    """Returns the positional argument object for name.

    Args:
      name: The Namespace metavar or destination.

    Raises:
      UnknownDestinationException: If there is no registered positional arg
        for name.

    Returns:
      The positional argument object for name.
    """
    dest = name.replace('-', '_').lower()
    meta = name.replace('-', '_').upper()
    for arg in self._GetCommand().ai.positional_args:
        if isinstance(arg, type):
            continue
        if dest == arg.dest or meta == arg.metavar:
            return arg
    raise parser_errors.UnknownDestinationException('No registered positional arg for [{}].'.format(name))